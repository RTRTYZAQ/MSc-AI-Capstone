#!/usr/bin/env python3
"""
Gradio Web Interface for Skin Disease Diagnosis
Combines EfficientNetV2-S classification with Qwen-VL multimodal analysis
"""

import json
import os
import tempfile
from typing import Dict, List, Tuple

import gradio as gr
import torch
from PIL import Image
import dashscope
from dashscope import MultiModalConversation
from single_image_inference import EfficientNetV2Classifier, get_transforms, find_latest_model_dir

# Set up DashScope API (make sure to set your API key)
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')


class SkinDiagnosisSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.transform = get_transforms(384)
        self.confidence_threshold = 0.3
        
        # Conversation state
        self.current_image_path = None
        self.current_predictions = []
        self.conversation_history = []
        
        # Load model and class names
        self._load_model()
        self._load_class_names()
    
    def _load_model(self):
        """Load the trained EfficientNetV2 model"""
        try:
            model_dir = find_latest_model_dir()
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
        """Load class names from category mapping"""
        try:
            category_mapping_path = os.path.join('processed_data', 'category_mapping.json')
            with open(category_mapping_path, 'r') as f:
                category_mapping = json.load(f)
            self.class_names = category_mapping['categories']
            print(f"Loaded {len(self.class_names)} class names")
            
        except Exception as e:
            print(f"Error loading class names: {e}")
            # Fallback class names if file not found
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
            prediction_context = "\\n".join([
                f"- {p['class_name']}: {p['confidence']:.3f} confidence ({p['confidence']*100:.1f}%)"
                for p in high_conf_predictions
            ])
            
            # System prompt for diagnosis report
            system_prompt = f"""You are an AI dermatology assistant. Based on the uploaded skin image and the AI classification results provided, generate a comprehensive diagnosis report.

IMPORTANT RULES:
1. Only consider conditions with confidence > {self.confidence_threshold}
2. If no conditions meet this threshold, do not generate a diagnosis
3. It is possible to recommend a treatment plan, including the required medications and precautions, but it should be emphasized that specific plans need to be consulted with professional physicians.
4. Use clear, professional medical language
5. Structure the report with the following sections:
   - PRELIMINARY ANALYSIS
   - MOST LIKELY CONDITIONS
   - CLINICAL OBSERVATIONS (Identify the affected body parts first)
   - RECOMMENDATIONS

Format the report in markdown for better readability."""
            
            user_prompt = f"""Please analyze this skin image and provide a diagnosis report based on the following AI classification results:

CLASSIFICATION RESULTS (Confidence > {self.confidence_threshold}):
{prediction_context}

Please provide a comprehensive diagnosis report following the specified format and rules."""
            
            # Call Qwen-VL API
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': [
                        {'image': image_path},
                        {'text': user_prompt}
                    ]
                }
            ]
            
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=messages,
                top_p=0.8,
                temperature=0.7
            )
            
            if response.status_code == 200:
                # Use the formatting function to handle response
                formatted_response = self._format_ai_response(response.output.choices[0].message.content)
                return formatted_response
            else:
                return str(f"❌ **Error generating report**: {response.message}")
                
        except Exception as e:
            # Provide a structured fallback report when API fails
            if "API" in str(e) or "DASHSCOPE" in str(e) or "network" in str(e).lower():
                return self._generate_fallback_report(predictions)
            else:
                return str(f"❌ **Error generating diagnosis report**: {str(e)}")
    
    def continue_conversation(self, user_message: str) -> str:
        """Continue multi-turn conversation about the diagnosis"""
        try:
            if not self.current_image_path or not self.current_predictions:
                return "⚠️ Please analyze an image first before starting a conversation."
            
            # Filter high-confidence predictions for context
            high_conf_predictions = [p for p in self.current_predictions if p['confidence'] > self.confidence_threshold]
            
            if not high_conf_predictions:
                return "⚠️ No high-confidence predictions available for detailed discussion."
            
            # Create context from predictions
            prediction_context = "\\n".join([
                f"- {p['class_name']}: {p['confidence']:.3f} confidence ({p['confidence']*100:.1f}%)"
                for p in high_conf_predictions
            ])
            
            # System prompt for follow-up conversation
            system_prompt = f"""You are an AI dermatology assistant. You can see the skin image in every message and have already analyzed it with the following results:

PREVIOUS ANALYSIS RESULTS:
{prediction_context}

The user is now asking follow-up questions about this diagnosis. You can refer to both the image and the previous analysis. Please provide helpful, accurate medical information while:

1. Always emphasizing the need for professional medical consultation
2. Being informative but not prescriptive about treatments
3. Focusing on education and general guidance
4. Maintaining context from the original analysis AND the visible image
5. You can describe what you see in the image to support your explanations
6. If asked about treatment, always recommend consulting a healthcare provider first

Keep responses concise but informative. Use markdown formatting for better readability."""
            
            # Build conversation history for context
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                }
            ]
            
            # Add conversation history
            for msg in self.conversation_history[-6:]:  # Keep last 6 messages for context
                messages.append(msg)
            
            # Add current user message with image context
            messages.append({
                'role': 'user',
                'content': [
                    {'image': self.current_image_path},  # 🔥 保持图片上下文
                    {'text': user_message}
                ]
            })
            
            # Call Qwen-VL API
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=messages,
                top_p=0.8,
                temperature=0.7
            )
            
            if response.status_code == 200:
                # Use the formatting function to handle response
                ai_response = self._format_ai_response(response.output.choices[0].message.content)
                
                # Update conversation history (保存用户消息时包含图片上下文)
                self.conversation_history.append({
                    'role': 'user',
                    'content': [
                        {'image': self.current_image_path},
                        {'text': user_message}
                    ]
                })
                self.conversation_history.append({
                    'role': 'assistant', 
                    'content': ai_response
                })
                
                return ai_response
            else:
                return f"❌ **Error in conversation**: {response.message}"
                
        except Exception as e:
            return f"❌ **Error in conversation**: {str(e)}"
    
    def reset_conversation(self):
        """Reset conversation state for new image analysis"""
        # Clean up previous temporary image file
        if self.current_image_path and os.path.exists(self.current_image_path):
            try:
                os.unlink(self.current_image_path)
                print(f"Cleaned up previous temporary file: {self.current_image_path}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary file: {e}")
        
        self.current_image_path = None
        self.current_predictions = []
        self.conversation_history = []
    
    def __del__(self):
        """Cleanup temporary files when object is destroyed"""
        if hasattr(self, 'current_image_path') and self.current_image_path and os.path.exists(self.current_image_path):
            try:
                os.unlink(self.current_image_path)
            except:
                pass  # Ignore errors during cleanup
    
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
                # Update diagnosis system state for conversation
                diagnosis_system.reset_conversation()  # Reset for new analysis
                diagnosis_system.current_image_path = tmp_path
                diagnosis_system.current_predictions = predictions
                
                report = diagnosis_system.generate_diagnosis_report(tmp_path, predictions)
                status_message = "✅ Analysis completed successfully. You can now ask follow-up questions!"
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise e
        else:
            report = """⚠️ **No High-Confidence Diagnosis Available**

No skin condition predictions exceed the confidence threshold of 50%. This could indicate:

- The condition is not well represented in the training data
- The image quality may need improvement
- The condition might be rare or require specialized analysis

**Recommendations:**
- Please consult a dermatologist for professional evaluation
- Consider taking a clearer, well-lit photograph
- Ensure the affected area is clearly visible in the image

*This AI system is designed to assist, not replace, professional medical diagnosis.*"""
            status_message = "⚠️ Low confidence predictions - professional consultation recommended"
        
        return predictions_html, str(report), str(status_message)
        
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        return "", str(error_msg), str(error_msg)


def handle_conversation(message: str, chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """
    Handle multi-turn conversation about the diagnosis
    
    Args:
        message: User's message
        chat_history: List of (user_message, bot_response) tuples
    
    Returns:
        Tuple of (updated_chat_history, empty_string_for_textbox_reset)
    """
    if not message.strip():
        return chat_history, ""
    
    try:
        # Get AI response
        ai_response = diagnosis_system.continue_conversation(message.strip())
        
        # Update chat history
        chat_history.append((message.strip(), ai_response))
        
        return chat_history, ""  # Return empty string to clear the input textbox
        
    except Exception as e:
        error_response = f"❌ Error in conversation: {str(e)}"
        chat_history.append((message.strip(), error_response))
        return chat_history, ""


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
            <h1>🏥 AI Skin Disease Diagnosis & Consultation System</h1>
            <p style="font-size: 18px; color: #666;">
                Upload a skin image for AI-powered analysis and engage in multi-turn conversations with our AI dermatology assistant
            </p>
            <p style="font-size: 16px; color: #444; font-weight: 500;">
                🔬 EfficientNetV2-S Classification + 🤖 Qwen-VL Multimodal Analysis + 💬 Interactive Consultation
            </p>
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
                
                report_display = gr.Markdown(
                    label="Diagnosis Report",
                    value="The detailed diagnosis report will appear here after analysis."
                )
                
                # Multi-turn conversation section
                gr.HTML("<h3>💬 Ask Follow-up Questions</h3>")
                gr.HTML("<p style='color: #666; font-size: 14px;'>After getting your diagnosis report, you can ask follow-up questions about the condition, symptoms, care tips, etc.</p>")
                
                chatbot = gr.Chatbot(
                    label="Conversation with AI Dermatology Assistant",
                    height=300,
                    show_label=True,
                    container=True
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about symptoms, treatment options, prevention, etc...",
                        scale=4
                    )
                    
                    chat_send_btn = gr.Button(
                        "Send",
                        variant="secondary",
                        scale=1
                    )
                
                chat_clear_btn = gr.Button(
                    "Clear Conversation",
                    variant="secondary",
                    size="sm"
                )
        
        def analyze_and_reset_chat(image):
            """Analyze image and reset chat history"""
            # Reset chat when new image is analyzed
            result = process_image(image)
            return result + ([],)  # Add empty chat history
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_and_reset_chat,
            inputs=[image_input],
            outputs=[predictions_display, report_display, status_display, chatbot]
        )
        
        # Chat event handlers
        chat_send_btn.click(
            fn=handle_conversation,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_input.submit(
            fn=handle_conversation,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        
        chat_clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, chat_input]
        )
        
        # Instructions section
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <h3>ℹ️ How to Use</h3>
            <ol>
                <li><strong>Upload Image:</strong> Click on the upload area and select a clear skin image</li>
                <li><strong>Analyze:</strong> Click the "Analyze Image" button to start processing</li>
                <li><strong>Review Results:</strong> Check the classification confidence scores and diagnosis report</li>
                <li><strong>Ask Questions:</strong> Use the chat feature to ask follow-up questions about the diagnosis</li>
                <li><strong>Medical Consultation:</strong> Always consult healthcare professionals for final diagnosis</li>
            </ol>
            
            <h4>📋 System Features</h4>
            <ul>
                <li>🔬 <strong>EfficientNetV2-S Classification:</strong> Trained on 23 skin condition categories</li>
                <li>🤖 <strong>Qwen-VL Multimodal Analysis:</strong> Advanced AI vision and reasoning</li>
                <li>📊 <strong>Confidence Thresholding:</strong> Only conditions with >50% confidence generate reports</li>
                <li>📝 <strong>Structured Reports:</strong> Professional medical format with recommendations</li>
                <li>💬 <strong>Multi-turn Conversations:</strong> Ask follow-up questions about diagnosis, symptoms, care tips</li>
            </ul>
            
            <h4>💡 Example Questions You Can Ask</h4>
            <ul>
                <li>"What are the typical symptoms of this condition?"</li>
                <li>"How can I prevent this condition from worsening?"</li>
                <li>"What should I avoid while dealing with this skin issue?"</li>
                <li>"When should I see a dermatologist urgently?"</li>
                <li>"Are there any home care tips for this condition?"</li>
                <li>"What are the possible complications if left untreated?"</li>
            </ul>
            
            <p style="color: #dc3545; font-weight: bold; margin-top: 15px;">
                ⚠️ Disclaimer: This AI system is for educational and research purposes only. 
                It should not be used as a substitute for professional medical diagnosis or treatment.
                The AI assistant provides general information only and cannot replace professional medical advice.
            </p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get('DASHSCOPE_API_KEY'):
        print("⚠️ Warning: DASHSCOPE_API_KEY environment variable not set.")
        print("Please set your DashScope API key to use Qwen-VL features.")
        print("The system will still work for EfficientNet classification only.")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=True,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed error messages
    )
