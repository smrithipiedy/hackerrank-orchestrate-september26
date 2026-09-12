import os
import json
import logging
from typing import Optional, Tuple
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

# Try to import AI libraries, fallback to mocks for environments without them
try:
    import easyocr
    from google import genai
    from google.genai import types
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ReceiptExtraction(BaseModel):
    """Structured schema for receipt amount extraction."""
    total_amount: float = Field(..., description="The final total amount of the transaction")
    currency: str = Field(..., description="The currency code (e.g., USD, ZAR, INR)")
    date: Optional[str] = Field(None, description="The date of the transaction in YYYY-MM-DD format")

class OCRVision:
    """
    Handles multimodal extraction of financial amounts from images.
    Implements a tiered approach: Gemini 2.5 Flash (Free) -> EasyOCR/Local Fallback.
    """

    def __init__(self, cache_path: str = ".cache/ocr_cache.json", api_key: Optional[str] = None):
        self.cache_path = cache_path
        self.api_key = api_key
        self.cache = self._load_cache()
        self.reader = None # Lazy load EasyOCR

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load OCR cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save OCR cache: {e}")

    def _get_local_fallback(self, image_path: str) -> Tuple[Optional[float], Optional[str]]:
        """Deterministic fallback using EasyOCR to find numbers."""
        if not HAS_AI_LIBS:
            logger.warning("AI libraries not installed. Cannot perform local fallback.")
            return None, None

        try:
            if self.reader is None:
                self.reader = easyocr.Reader(['en'])

            results = self.reader.readtext(image_path)
            # Simple heuristic: look for the largest number that looks like a total
            numbers = []
            for (bbox, text, prob) in results:
                # Clean text to keep only digits and decimal point
                clean_text = "".join(c for c in text if c.isdigit() or c == '.')
                try:
                    if clean_text:
                        numbers.append(float(clean_text))
                except ValueError:
                    continue

            if numbers:
                # Heuristic: The total is usually one of the largest numbers on a receipt
                return max(numbers), "Unknown"
            return None, None
        except Exception as e:
            logger.error(f"EasyOCR fallback failed: {e}")
            return None, None

    def extract_amount(self, image_path: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extracts total amount and currency from an image.
        Priority: Cache -> Gemini -> Local Fallback.
        """
        # 1. Check Cache
        if image_path in self.cache:
            cached = self.cache[image_path]
            return cached.get('total_amount'), cached.get('currency')

        # 2. Primary Extraction: Gemini 2.5 Flash
        if self.api_key and HAS_AI_LIBS:
            try:
                client = genai.Client(api_key=self.api_key)
                img = Image.open(image_path)

                prompt = (
                    "Extract the total amount and currency from this receipt. "
                    "Return ONLY a JSON object matching this schema: "
                    '{"total_amount": float, "currency": "string", "date": "YYYY-MM-DD or null"}'
                )

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt, img],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ReceiptExtraction,
                    ),
                )

                data = ReceiptExtraction.model_validate_json(response.text)
                self.cache[image_path] = data.model_dump()
                self._save_cache()
                return data.total_amount, data.currency
            except Exception as e:
                logger.warning(f"Gemini extraction failed for {image_path}: {e}. Trying fallback...")

        # 3. Local Fallback: EasyOCR
        amount, currency = self._get_local_fallback(image_path)
        if amount is not None:
            self.cache[image_path] = {"total_amount": amount, "currency": currency}
            self._save_cache()
            return amount, currency

        logger.error(f"All extraction methods failed for {image_path}")
        return None, None
