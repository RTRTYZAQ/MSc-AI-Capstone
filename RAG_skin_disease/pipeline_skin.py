import os
# 启用更快的下载器（如已安装 hf_transfer 包）
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
# 默认使用 Hugging Face 镜像（可通过外部设置 HF_ENDPOINT 覆盖）
hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_ENDPOINT', hf_endpoint)

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from pyprojroot import here

from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestorBGE


@dataclass
class SkinPipelinePaths:
    """Folder layout for the skin dataset processing pipeline."""

    root: Path
    pdf_root: Path
    parsed_root: Path
    merged_root: Path
    markdown_root: Path
    chunked_root: Path
    vector_root: Path

    @classmethod
    def from_root(cls, root: Path) -> "SkinPipelinePaths":
        pdf_root = root / "RAG_pdfs"
        parsed_root = root / "parsed_reports"
        merged_root = root / "merged_reports"
        markdown_root = root / "reports_markdown"
        chunked_root = root / "chunked_reports"
        vector_root = root / "vector_dbs_bge"
        return cls(
            root=root,
            pdf_root=pdf_root,
            parsed_root=parsed_root,
            merged_root=merged_root,
            markdown_root=markdown_root,
            chunked_root=chunked_root,
            vector_root=vector_root,
        )


class SkinPipeline:
    """Processing pipeline dedicated to the skin_set dataset."""

    def __init__(self, root_path: Path, use_serialized_tables: bool = False):
        self.paths = SkinPipelinePaths.from_root(root_path)
        self.use_serialized_tables = use_serialized_tables

    def _group_pdfs_by_category(self) -> Dict[Path, List[Path]]:
        """Return mapping of category-relative folders to PDF lists."""
        pdf_groups: Dict[Path, List[Path]] = defaultdict(list)

        if not self.paths.pdf_root.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.paths.pdf_root}")

        for pdf_path in self.paths.pdf_root.rglob("*.pdf"):
            if not pdf_path.is_file():
                continue
            relative_dir = pdf_path.parent.relative_to(self.paths.pdf_root)
            pdf_groups[relative_dir].append(pdf_path)

        return pdf_groups

    @staticmethod
    def _iter_category_dirs(root: Path, pattern: str) -> Iterable[Tuple[Path, Path]]:
        """Yield (relative_path, directory) pairs containing files matching pattern."""
        seen: Set[Path] = set()
        for file_path in root.rglob(pattern):
            category_dir = file_path.parent
            relative_dir = category_dir.relative_to(root)
            if relative_dir in seen:
                continue
            seen.add(relative_dir)
            yield relative_dir, category_dir

    def parse_pdf_reports(self) -> None:
        """Parse PDFs into structured JSON while preserving folder hierarchy."""
        pdf_groups = self._group_pdfs_by_category()
        if not pdf_groups:
            print("No PDF files found to parse.")
            return

        for relative_dir, pdf_paths in sorted(pdf_groups.items(), key=lambda item: str(item[0])):
            output_dir = self.paths.parsed_root / relative_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            parser = PDFParser(output_dir=output_dir)
            parser.parse_and_export(input_doc_paths=pdf_paths)
            print(f"Parsed {len(pdf_paths)} PDFs into {output_dir}")

    def merge_reports(self) -> None:
        """Normalize parsed reports into per-page JSON files per category."""
        ptp = PageTextPreparation(use_serialized_tables=self.use_serialized_tables)

        for relative_dir, category_dir in self._iter_category_dirs(self.paths.parsed_root, "*.json"):
            output_dir = self.paths.merged_root / relative_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            ptp.process_reports(reports_dir=category_dir, output_dir=output_dir)
            print(f"Merged reports saved to {output_dir}")

    def export_reports_to_markdown(self) -> None:
        """Generate markdown exports matching the PDF folder structure."""
        ptp = PageTextPreparation(use_serialized_tables=self.use_serialized_tables)

        for relative_dir, category_dir in self._iter_category_dirs(self.paths.parsed_root, "*.json"):
            output_dir = self.paths.markdown_root / relative_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            ptp.export_to_markdown(reports_dir=category_dir, output_dir=output_dir)
            print(f"Markdown reports written to {output_dir}")

    def chunk_reports(self) -> None:
        """Split merged reports into chunks per category for downstream ingestion."""
        splitter = TextSplitter()

        for relative_dir, category_dir in self._iter_category_dirs(self.paths.merged_root, "*.json"):
            output_dir = self.paths.chunked_root / relative_dir
            splitter.split_all_reports(
                all_report_dir=category_dir,
                output_dir=output_dir,
                serialized_tables_dir=(self.paths.parsed_root / relative_dir) if self.use_serialized_tables else None,
            )
            print(f"Chunked reports stored at {output_dir}")

    def create_vector_dbs(self, batch_size: int = 12, max_length: int = 8192) -> None:
        """Build FAISS vector indices for each category's chunked reports."""
        ingestor = VectorDBIngestorBGE(model_name="BAAI/bge-m3", use_fp16=True)
        ingest_kwargs = {"batch_size": batch_size, "max_length": max_length}

        processed = False
        for relative_dir, category_dir in self._iter_category_dirs(self.paths.chunked_root, "*.json"):
            output_dir = self.paths.vector_root / relative_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            ingestor.process_reports(category_dir, output_dir, **ingest_kwargs)

            print(f"Vector databases created in {output_dir}")
            processed = True

        if not processed:
            print("No chunked reports found for vector database creation.")

    def run(self) -> None:
        """Execute all steps sequentially."""
        print("1. Parsing PDF reports...")
        # self.parse_pdf_reports()

        print("2. Merging parsed reports...")
        # self.merge_reports()

        print("3. Exporting markdown reports...")
        # self.export_reports_to_markdown()

        print("4. Chunking reports...")
        # self.chunk_reports()

        print("5. Creating vector databases...")
        self.create_vector_dbs()

        print("Skin pipeline completed.")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    skin_root = script_dir / "data" / "skin_set"
    pipeline = SkinPipeline(root_path=skin_root)
    pipeline.run()
