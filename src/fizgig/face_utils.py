"""
Face Detection Utilities for Fizgig
====================================
Uses InsightFace for face detection and gender classification.
Designed to work with CPU-only inference for GPU-agnostic operation.
"""

import os
from typing import List, Tuple, Optional, NamedTuple
from PIL import Image
import numpy as np

# Suppress TensorFlow/ONNX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FaceInfo(NamedTuple):
    """Information about a detected face"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    gender: str  # 'male' or 'female'
    gender_score: float  # Confidence (0-1)
    area: int  # Face area in pixels
    center: Tuple[int, int]  # Center point (x, y)


class FaceDetector:
    """
    Face detector using InsightFace with lazy loading.

    Uses CPU-only inference for GPU-agnostic operation.
    Models are downloaded automatically on first use (~300MB).

    Usage:
        detector = FaceDetector()
        faces = detector.detect_all("image.jpg")
        largest = detector.get_largest(faces)
        male_face = detector.get_largest_by_gender(faces, "male")
    """

    def __init__(self):
        self._app = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if InsightFace is available"""
        if self._available is None:
            try:
                import insightface
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _ensure_loaded(self):
        """Lazy load the FaceAnalysis model"""
        if self._app is not None:
            return

        if not self.available:
            raise ImportError(
                "InsightFace is not installed. "
                "Run install_fizgig.py to set up face detection."
            )

        from insightface.app import FaceAnalysis

        # Initialize with detection and gender/age modules
        self._app = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=['detection', 'genderage'],
            providers=['CPUExecutionProvider']
        )
        # ctx_id=-1 forces CPU usage
        self._app.prepare(ctx_id=-1)

    def detect_all(self, image_path: str) -> List[FaceInfo]:
        """
        Detect all faces in an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of FaceInfo objects for each detected face
        """
        self._ensure_loaded()

        # Load image with OpenCV (InsightFace expects BGR format)
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run detection
        faces = self._app.get(img)

        results = []
        for face in faces:
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Get gender (0=female, 1=male)
            gender_idx = getattr(face, 'gender', None)
            if gender_idx is not None:
                gender = 'male' if gender_idx == 1 else 'female'
                # InsightFace doesn't provide gender confidence directly,
                # but we can use detection score as a proxy
                gender_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
            else:
                gender = 'unknown'
                gender_score = 0.0

            # Calculate area and center
            area = (x2 - x1) * (y2 - y1)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            results.append(FaceInfo(
                bbox=(x1, y1, x2, y2),
                gender=gender,
                gender_score=gender_score,
                area=area,
                center=center
            ))

        return results

    def detect_from_pil(self, pil_image: Image.Image) -> List[FaceInfo]:
        """
        Detect faces from a PIL Image object.

        Args:
            pil_image: PIL Image object

        Returns:
            List of FaceInfo objects
        """
        self._ensure_loaded()

        import cv2

        # Convert PIL to numpy array (RGB)
        img_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV/InsightFace
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA - convert to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            # Grayscale - convert to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Run detection
        faces = self._app.get(img_bgr)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            gender_idx = getattr(face, 'gender', None)
            if gender_idx is not None:
                gender = 'male' if gender_idx == 1 else 'female'
                gender_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
            else:
                gender = 'unknown'
                gender_score = 0.0

            area = (x2 - x1) * (y2 - y1)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            results.append(FaceInfo(
                bbox=(x1, y1, x2, y2),
                gender=gender,
                gender_score=gender_score,
                area=area,
                center=center
            ))

        return results

    def get_largest(self, faces: List[FaceInfo]) -> Optional[FaceInfo]:
        """
        Get the largest face from a list of detected faces.

        Args:
            faces: List of FaceInfo objects

        Returns:
            The FaceInfo with the largest area, or None if no faces
        """
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    def get_largest_by_gender(
        self,
        faces: List[FaceInfo],
        gender: str,
        fallback_to_any: bool = True
    ) -> Optional[FaceInfo]:
        """
        Get the largest face of a specific gender.

        Args:
            faces: List of FaceInfo objects
            gender: 'male' or 'female'
            fallback_to_any: If True and no matching gender found,
                           return the largest face of any gender

        Returns:
            The largest FaceInfo matching the gender, or None
        """
        gender = gender.lower()
        matching = [f for f in faces if f.gender == gender]

        if matching:
            return max(matching, key=lambda f: f.area)

        if fallback_to_any and faces:
            return self.get_largest(faces)

        return None


def crop_to_face(
    image: Image.Image,
    face: FaceInfo,
    padding_percent: float = 20.0,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Crop an image to a detected face with padding.

    Args:
        image: PIL Image to crop
        face: FaceInfo object with bounding box
        padding_percent: Extra space around face as percentage (0-100)
        maintain_aspect: If True, make the crop region square

    Returns:
        Cropped PIL Image
    """
    x1, y1, x2, y2 = face.bbox
    face_w = x2 - x1
    face_h = y2 - y1

    # Calculate padding
    pad_w = int(face_w * padding_percent / 100)
    pad_h = int(face_h * padding_percent / 100)

    # Apply padding
    crop_x1 = x1 - pad_w
    crop_y1 = y1 - pad_h
    crop_x2 = x2 + pad_w
    crop_y2 = y2 + pad_h

    # Make square if requested
    if maintain_aspect:
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w > crop_h:
            # Wider than tall - expand height
            diff = crop_w - crop_h
            crop_y1 -= diff // 2
            crop_y2 += diff - diff // 2
        elif crop_h > crop_w:
            # Taller than wide - expand width
            diff = crop_h - crop_w
            crop_x1 -= diff // 2
            crop_x2 += diff - diff // 2

    # Clamp to image bounds
    img_w, img_h = image.size
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)

    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def draw_face_boxes(
    image: Image.Image,
    faces: List[FaceInfo],
    highlight_index: Optional[int] = None
) -> Image.Image:
    """
    Draw bounding boxes on faces for preview.

    Args:
        image: PIL Image to draw on (will be copied)
        faces: List of FaceInfo objects
        highlight_index: Index of face to highlight in green (others yellow)

    Returns:
        New PIL Image with boxes drawn
    """
    from PIL import ImageDraw, ImageFont

    # Make a copy to draw on
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    # Try to get a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox

        # Color: green for highlighted, yellow for others
        if highlight_index is not None and i == highlight_index:
            color = (0, 255, 0)  # Green
            width = 3
        else:
            color = (255, 255, 0)  # Yellow
            width = 2

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # Draw label
        gender_char = 'M' if face.gender == 'male' else 'F' if face.gender == 'female' else '?'
        label = f"{gender_char} ({face.area:,}px)"

        # Calculate label position
        label_y = y1 - 20 if y1 > 20 else y2 + 5

        # Draw label background
        bbox = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((x1, label_y), label, fill=color, font=font)

    return preview


class FaceEmbedder:
    """
    Face-embedding extractor (ArcFace via InsightFace) for look-consistency scoring.

    Separate from FaceDetector because it loads the recognition module (detection +
    recognition) — FaceDetector only loads detection + genderage. Same lazy-loading,
    CPU-only pattern. Embeddings are 512-dim and L2-normalized (`normed_embedding`),
    so cosine similarity between two faces is a plain dot product.

    Usage:
        embedder = FaceEmbedder()
        emb = embedder.embed("image.png")     # largest face's embedding, or None
        sim = float(np.dot(emb_a, emb_b))     # cosine similarity
    """

    def __init__(self):
        self._app = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import insightface  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _ensure_loaded(self):
        if self._app is not None:
            return
        if not self.available:
            raise ImportError(
                "InsightFace is not installed. "
                "Run install_fizgig.py to set up face detection."
            )
        from insightface.app import FaceAnalysis
        self._app = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=['detection', 'recognition'],
            providers=['CPUExecutionProvider']
        )
        self._app.prepare(ctx_id=-1)

    def _detect_with_pad_retry(self, img_bgr):
        """Detect faces, retrying with a padded border if none are found.

        RetinaFace (buffalo_l) fails on faces that FILL the frame — a close-up
        headshot is too large for its anchor scales, so it returns nothing.
        Padding a border around the image shrinks the face's fraction of the
        frame, bringing it back into the detector's working range; it shifts
        coordinates but not the face content, so the embedding is unchanged.
        Only pads when the unpadded pass fails (no regression for normal
        framing). Same fix as src/fizgig/lora_royale/likeness.py.
        """
        import cv2
        faces = self._app.get(img_bgr)
        if faces:
            return faces
        for pad in (0.25, 0.5):
            h, w = img_bgr.shape[:2]
            py, px = int(h * pad), int(w * pad)
            padded = cv2.copyMakeBorder(img_bgr, py, py, px, px, cv2.BORDER_REPLICATE)
            faces = self._app.get(padded)
            if faces:
                return faces
        return []

    def embed(self, image_path: str) -> Optional[np.ndarray]:
        """
        Embedding of the LARGEST detected face in the image, or None if no face.

        Loads via PIL (handles unicode paths on Windows, unlike cv2.imread) and
        converts to the BGR array InsightFace expects.
        """
        self._ensure_loaded()
        import cv2
        pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        faces = self._detect_with_pad_retry(img_bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = getattr(largest, "normed_embedding", None)
        return None if emb is None else np.asarray(emb, dtype=np.float32)


# Convenience function for checking availability
def is_face_detection_available() -> bool:
    """Check if face detection dependencies are installed"""
    try:
        import insightface
        import cv2
        import onnxruntime
        return True
    except ImportError:
        return False
