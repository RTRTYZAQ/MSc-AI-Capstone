import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
### 这里用了国内镜像，不需要的话可以注释掉
hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import gradio as gr
import torch
from PIL import Image
import dashscope
from dashscope import MultiModalConversation
from src.single_image_inference import EfficientNetV2Classifier, get_transforms
from src.bge_retrieval import BGERetrieval

# Set up DashScope API (make sure to set your API key)
### 可以删掉
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')


class SkinDiagnosisSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.transform = get_transforms(384)
        self.confidence_threshold = 0.3
        self.base_dir = Path(__file__).resolve().parent
        self.rag_root = self.base_dir / 'data' / 'skin_set'
        self.retrieval: Optional[BGERetrieval] = None
        self.retrieval_min_score = 0.5
        self.retrieval_max_chunks = 2
        self.retrieval_topk = 6
        self._knowledge_cache: Dict[str, str] = {}
        
        # Load model and class names
        self._load_model()
        self._load_class_names()
        self._init_retrieval()
    
    def _load_model(self):
        """Load the trained EfficientNetV2 model"""
        try:
            model_dir = "models\EfficientNetV2-M_20250911_224153"
            model_path = os.path.join(model_dir, 'best_model_efficientnet_v2.pth')
            
            self.model = EfficientNetV2Classifier(num_classes=23)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully from: {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_class_names(self):
        self.class_names = [
            'Acne and Rosacea Photos',
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            'Atopic Dermatitis Photos',
            'Bullous Disease Photos',
            'Cellulitis Impetigo and other Bacterial Infections',
            'Eczema Photos',
            'Exanthems and Drug Eruptions',
            'Hair Loss Photos Alopecia and other Hair Diseases',
            'Herpes HPV and other STDs Photos',
            'Light Diseases and Disorders of Pigmentation',
            'Lupus and other Connective Tissue diseases',
            'Melanoma Skin Cancer Nevi and Moles',
            'Nail Fungus and other Nail Disease',
            'Poison Ivy Photos and other Contact Dermatitis',
            'Psoriasis pictures Lichen Planus and related diseases',
            'Scabies Lyme Disease and other Infestations and Bites',
            'Seborrheic Keratoses and other Benign Tumors',
            'Systemic Disease',
            'Tinea Ringworm Candidiasis and other Fungal Infections',
            'Urticaria Hives',
            'Vascular Tumors',
            'Vasculitis Photos',
            'Warts Molluscum and other Viral Infections'
        ]

    def _init_retrieval(self) -> None:
        """Initialize the RAG retriever if resources exist."""
        try:
            if not self.rag_root.exists():
                print(f"RAG resources not found at {self.rag_root}. RAG will be disabled.")
                return
            self.retrieval = BGERetrieval(
                root_path=self.rag_root,
                use_reranker=True,
                rerank_candidates=max(self.retrieval_topk, self.retrieval_max_chunks * 3),
            )
            print("RAG retrieval initialized successfully.")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to initialize RAG retrieval: {exc}")
            self.retrieval = None

    @staticmethod
    def _normalize_category(name: str) -> str:
        category = name.replace('_', ' ').strip()
        if category.lower().endswith(' photos'):
            category = category[: -len(' photos')].rstrip()
        return category

    def _gather_chunks(self, question: str) -> List[Tuple[float, Dict[str, object]]]:
        if not self.retrieval:
            return []
        results = self.retrieval.retrieve(question, self.retrieval_topk)
        if not results:
            return []
        filtered = [(score, chunk) for score, chunk in results if score >= self.retrieval_min_score]
        if not filtered:
            filtered = results[:1]
        return filtered[: self.retrieval_max_chunks]

    def _build_relevant_knowledge(self, category: str) -> str:
        if not self.retrieval:
            return json.dumps({'disease': [], 'treatment': []})

        cached = self._knowledge_cache.get(category)
        if cached is not None:
            return cached

        disease_question = f"What is {category}?"
        treatment_question = f"What are the recommended treatment options for {category}?"
        disease_chunks = self._gather_chunks(disease_question)
        treatment_chunks = self._gather_chunks(treatment_question)

        payload = {
            'disease': [
                {
                    'question': disease_question,
                    'confidence': round(float(score), 4),
                    'text': ' '.join(str(chunk.get('text', '')).split()),
                }
                for score, chunk in disease_chunks
            ],
            'treatment': [
                {
                    'question': treatment_question,
                    'confidence': round(float(score), 4),
                    'text': ' '.join(str(chunk.get('text', '')).split()),
                }
                for score, chunk in treatment_chunks
            ],
        }

        knowledge = json.dumps(payload, ensure_ascii=False, indent=2)
        self._knowledge_cache[category] = knowledge
        return knowledge

    def _build_llava_system_prompt(self, category: str, relevant_knowledge: str) -> str:
        return (
            "<image>\n"
            f"The image is from the dermatology category: {category}.\n"
            "Please generate a diagnostic report based on the skin images uploaded by the patient and the relevant knowledge about the skin condition provided. \n"
            f"Relevant knowledge: {relevant_knowledge}\n"
            "Please output strictly in JSON format, without any extra explanation:\n"
            "{\n"
            '  "Disease Name": "<predicted disease name>",\n'
            '  "Symptom Description": "<simple description of patient\'s skin condition symptoms>",\n'
            '  "Treatment Plan Recommendation": "<given this patient\'s skin condition and the relevant knowledge about this dermatological disorder, please provide your treatment plan recommendation.>"\n'
            "}"
        )

    @staticmethod
    def _json_report_to_markdown(report: Dict[str, object]) -> str:
        disease = str(report.get('Disease Name', 'N/A')).strip()
        symptoms = str(report.get('Symptom Description', 'N/A')).strip()
        treatment = str(report.get('Treatment Plan Recommendation', 'N/A')).strip()
        return (
            "#### Disease Name\n"
            f"- {disease}\n\n"
            "#### Symptom Description\n"
            f"{symptoms}\n\n"
            "#### Treatment Plan Recommendation\n"
            f"{treatment}"
        )

    def _generate_fallback_report(self, predictions: List[Dict]) -> str:
        if not predictions:
            return "⚠️ **Unable to generate report**\n\nThe assistant could not produce a diagnosis."
        top_prediction = predictions[0]
        normalized_category = self._normalize_category(top_prediction['class_name'])
        knowledge = self._build_relevant_knowledge(normalized_category)
        fallback = (
            "**Diagnosis Service Unavailable**\n\n"
            "The language model could not be reached. Showing the top prediction and retrieved references instead.\n\n"
            "### Top Prediction\n"
            f"- Condition: **{top_prediction['class_name']}**\n"
            f"- Confidence: **{top_prediction['confidence']*100:.1f}%**\n\n"
            "### Retrieved Knowledge\n"
            f"```json\n{knowledge}\n```"
        )
        return fallback
    
    def predict_image(self, image: Image.Image, top_k: int = 5) -> Dict:
        """Predict skin disease for a single image"""
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0)
            
            # Get top-k predictions
            top_k = min(top_k, len(self.class_names))
            topk_probs, topk_indices = torch.topk(probs, top_k)
            
            # Convert to numpy
            topk_probs = topk_probs.cpu().numpy()
            topk_indices = topk_indices.cpu().numpy()
            
            # Create results
            predictions = []
            confidence_dict = {}
            
            for i, (idx, prob) in enumerate(zip(topk_indices, topk_probs)):
                class_name = self.class_names[idx]
                confidence = float(prob)
                
                predictions.append({
                    'rank': i + 1,
                    'class_name': class_name,
                    'confidence': confidence
                })
                confidence_dict[class_name] = confidence
            
            return {
                'predictions': predictions,
                'confidence_dict': confidence_dict,
                'has_high_confidence': any(p['confidence'] > self.confidence_threshold for p in predictions)
            }
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
    
    def generate_diagnosis_report(self, image_path: str, predictions: List[Dict]) -> str:
        """Generate diagnosis report using Qwen-VL"""
        try:
            # Filter high-confidence predictions
            high_conf_predictions = [p for p in predictions if p['confidence'] > self.confidence_threshold]
            
            if not high_conf_predictions:
                return str(f"⚠️ **No High-Confidence Diagnosis Available**\\n\\nNo skin condition predictions exceed the confidence threshold of {self.confidence_threshold}. Please consult a dermatologist for professional evaluation.")

            # Create context from high-confidence predictions
            prediction_context = "\n".join([
                f"- {p['class_name']}: {p['confidence']:.3f} confidence ({p['confidence']*100:.1f}%)"
                for p in high_conf_predictions
            ])

            top_prediction = predictions[0]
            normalized_category = self._normalize_category(top_prediction['class_name'])
            relevant_knowledge = self._build_relevant_knowledge(normalized_category)
            system_prompt = self._build_llava_system_prompt(normalized_category, relevant_knowledge)
            ### messages就是上下文，包括图片临时路径和prompt
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'image': image_path},
                        {'text': system_prompt}
                    ]
                }
            ]
            ### 调用模型，这里的messages只是一种传参方法，可以单独传下面两个变量
            ### image_path=messages[0]['content'][0]['image']
            ### prompt=messages[0]['content'][1]['text']
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=messages,
                top_p=0.8,
                temperature=0.7
            )
            
            formatted_response = self._format_ai_response(response.output.choices[0].message.content)
            formatted_response = formatted_response.removeprefix("```json").removesuffix("```").strip()
            try:
                json_report = json.loads(formatted_response)
                return self._json_report_to_markdown(json_report)
            except json.JSONDecodeError:
                return formatted_response
                
        except Exception as e:
            # Provide a structured fallback report when API fails
            if "API" in str(e) or "DASHSCOPE" in str(e) or "network" in str(e).lower():
                return self._generate_fallback_report(predictions)
            else:
                return str(f"❌ **Error generating diagnosis report**: {str(e)}")
    
    
    
    def _format_ai_response(self, response_content) -> str:
        """Format AI response to ensure it's a clean text string"""
        try:
            if isinstance(response_content, list):
                # Handle list format response
                text_parts = []
                for item in response_content:
                    if isinstance(item, dict):
                        if 'text' in item:
                            text_parts.append(str(item['text']))
                        elif 'content' in item:
                            text_parts.append(str(item['content']))
                        else:
                            # Handle other dict structures
                            text_parts.append(str(item))
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                
                return "\n".join(text_parts).strip()
            
            elif isinstance(response_content, dict):
                # Handle dict format response
                if 'text' in response_content:
                    return str(response_content['text']).strip()
                elif 'content' in response_content:
                    return str(response_content['content']).strip()
                else:
                    return str(response_content).strip()
            
            else:
                # Handle string or other formats
                return str(response_content).strip()
                
        except Exception as e:
            return f"❌ Error formatting response: {str(e)}"


# Initialize the diagnosis system
diagnosis_system = SkinDiagnosisSystem()


def process_image(image: Image.Image) -> Tuple[str, str, str]:
    """
    Process uploaded image and return predictions and diagnosis report
    
    Returns:
        Tuple of (predictions_html, report_markdown, status_message)
    """
    if image is None:
        return "", str("Please upload an image first."), str("❌ No image provided")
    
    try:
        # Predict with EfficientNet
        result = diagnosis_system.predict_image(image, top_k=5)
        predictions = result['predictions']
        has_high_confidence = result['has_high_confidence']
        
        # Format predictions as HTML table
        predictions_html = """
        <div style="margin: 10px 0;">
            <h3>🔬 AI Classification Results</h3>
            <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                <thead>
                    <tr style="background-color: #f0f0f0;">
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Rank</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Condition</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Confidence</th>
                        <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Probability</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for pred in predictions:
            confidence_color = "green" if pred['confidence'] > 0.5 else "orange" if pred['confidence'] > 0.3 else "red"
            predictions_html += f"""
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;">{pred['rank']}</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{pred['class_name']}</td>
                        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {confidence_color}; font-weight: bold;">{pred['confidence']:.3f}</td>
                        <td style="border: 1px solid #ddd; padding: 8px; text-align: center; color: {confidence_color}; font-weight: bold;">{pred['confidence']*100:.1f}%</td>
                    </tr>
            """
        
        predictions_html += """
                </tbody>
            </table>
        </div>
        """
        
        # Generate diagnosis report if high confidence predictions exist
        if has_high_confidence:
            # Save image temporarily for Qwen-VL
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name
            
            try:
                report = diagnosis_system.generate_diagnosis_report(tmp_path, predictions)
                status_message = "✅ Analysis completed successfully."
            finally:
                # Always clean up temporary file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        else:
            report = """⚠️ **No High-Confidence Diagnosis Available**"""
            status_message = "⚠️ Low confidence predictions - professional consultation recommended"
        
        return predictions_html, str(report), str(status_message)
        
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        return "", str(error_msg), str(error_msg)

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .image-container {
        max-height: 500px !important;
    }
    """
    
    with gr.Blocks(css=css, title="AI Skin Disease Diagnosis System") as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1> AI Skin Disease Diagnosis & Consultation System</h1>
            <p style="font-size: 14px; color: #888; font-style: italic;">
                ⚠️ This system is for educational/research purposes only. Always consult healthcare professionals for medical advice.
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Upload Image</h3>")
                image_input = gr.Image(
                    type="pil",
                    label="Upload skin image",
                    elem_classes=["image-container"]
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
                
                status_display = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=2
                )
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📊 Analysis Results</h3>")
                
                predictions_display = gr.HTML(
                    label="Classification Results",
                    value="Upload an image and click 'Analyze Image' to see results."
                )
                
                gr.HTML("<h3>🩺 AI Diagnosis Report</h3>")

                report_display = gr.Markdown(
                    label="Diagnosis Report",
                    value="The detailed diagnosis report will appear here after analysis."
                )
                
        # Event handler (no chat functionality)
        analyze_btn.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[predictions_display, report_display, status_display]
        )
        
        # Instructions section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h3>How to Use</h3>
            <ol>
                <li><strong>Upload Image:</strong> Click on the upload area and select a clear skin image</li>
                <li><strong>Analyze:</strong> Click the "Analyze Image" button to start processing</li>
                <li><strong>Review Results:</strong> Check the classification confidence scores and diagnosis report</li>
                <li><strong>Medical Consultation:</strong> Always consult healthcare professionals for final diagnosis</li>
            </ol>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=True,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed error messages
    )
