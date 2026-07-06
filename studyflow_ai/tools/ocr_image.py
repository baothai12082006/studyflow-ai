# Image OCR Tool Wrapper
from google.adk import tool
from studyflow_ai.services.ocr_service import OCRService

@tool
async def ocr_image_tool(image_uri: str) -> dict:
    """Performs OCR on image or slides format documents by invoking the OCR Service.
    
    Args:
        image_uri: String location of target image
    """
    ocr_service = OCRService()
    return await ocr_service.extract_text(image_uri)
