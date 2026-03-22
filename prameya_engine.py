import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy.fftpack import dct
from PIL import ExifTags
import os
import re
from typing import Dict, List, Tuple


class PrameyaEngine:
    """
    Forensic framework for image authentication based on the
    "Six Pillars of Detection".

    Parameters
    ----------
    image_path : str
        Path to the input image to be analyzed.
    """

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.image = self._load_image(image_path)
        # Cached maps for visualization
        self._last_ela_map: np.ndarray | None = None
        self._last_svd_matches: List[Tuple[int, int, int, int]] = []

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray | None:
        """
        Load an image from disk using OpenCV (BGR format).
        Returns None if the image cannot be loaded.
        """
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return image

    # --- ELA core routine ---

    def run_ela(self, quality: int = 90, scale: float = 15.0) -> float:
        """
        Run Error Level Analysis (ELA) on the current image.

        Steps:
        - Load the image with Pillow.
        - Resave at the given JPEG quality.
        - Compute the pixel-wise absolute difference between the original
          and resaved versions.
        - Enhance the differences for visualization using a brightness scale.
        - Compute a normalized tamper score based on the variance of the
          difference map.

        Parameters
        ----------
        quality : int, optional
            JPEG quality used when resaving (default is 90).
        scale : float, optional
            Factor to enhance the ELA difference image (default is 15.0).

        Returns
        -------
        float
            Normalized tamper score in the range [0, 1], where higher
            values indicate stronger compression inconsistencies.
        """
        if self.image_path is None:
            raise ValueError("No image path provided for ELA.")

        # Open original image with Pillow (convert to RGB for consistency).
        original = Image.open(self.image_path).convert("RGB")

        # Resave the image at the specified JPEG quality into memory.
        from io import BytesIO

        buffer = BytesIO()
        original.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        # Compute pixel-wise absolute difference.
        diff = ImageChops.difference(original, resaved)

        # Enhance the differences to make artifacts more visible.
        enhancer = ImageEnhance.Brightness(diff)
        ela_image = enhancer.enhance(scale)

        # Convert to NumPy array (values in [0, 255]) for scoring and map caching.
        ela_array = np.asarray(ela_image).astype(np.float32)

        # Derive a single-channel tamper-intensity map:
        # brighter pixels indicate stronger local inconsistencies.
        if ela_array.ndim == 3:
            # Simple luminance-like projection.
            r = ela_array[:, :, 0]
            g = ela_array[:, :, 1]
            b = ela_array[:, :, 2]
            ela_gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            ela_gray = ela_array

        # Normalize to [0, 255] and cache for visualization.
        min_v = float(ela_gray.min())
        max_v = float(ela_gray.max())
        if max_v > min_v:
            norm = (ela_gray - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(ela_gray, dtype=np.float32)
        self._last_ela_map = (norm * 255.0).astype(np.uint8)

        # Global variance for the overall ELA tamper score.
        variance = float(np.var(ela_array))

        # Normalize variance to [0, 1].
        # The normalization constant here is heuristic; adjust as needed
        # based on empirical analysis of your dataset.
        # We assume that most natural images will have ELA variance well
        # below this value.
        normalization_const = 255.0**2 / 4.0  # heuristic upper bound
        tamper_score = variance / normalization_const
        tamper_score = float(np.clip(tamper_score, 0.0, 1.0))

        return tamper_score

    # --- Double Quantization (DQ) core routine ---

    def run_double_quantization(self, block_size: int = 8, bins: int = 50) -> float:
        """
        Run Double Quantization (DQ) analysis using 2D DCT over 8x8 blocks.

        Steps:
        - Convert the image to grayscale and to float32.
        - Split into non-overlapping 8x8 blocks.
        - Compute 2D DCT on each block (using scipy.fftpack.dct).
        - Collect AC coefficients (all except DC term).
        - Build a histogram of AC coefficients and look for periodic
          "holes" or zero-count bins indicative of double quantization.

        Parameters
        ----------
        block_size : int, optional
            Size of the blocks used for DCT (default is 8).
        bins : int, optional
            Number of histogram bins for AC coefficient magnitudes (default is 50).

        Returns
        -------
        float
            Normalized tamper score in the range [0, 1] based on the
            proportion and depth of zero-count bins in the histogram.
        """
        if self.image is None:
            raise ValueError("Image not loaded for DQ analysis.")

        # Convert to grayscale and float32.
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        h, w = gray.shape
        bs = block_size
        # Trim image so that it is divisible by block_size.
        h_trim = (h // bs) * bs
        w_trim = (w // bs) * bs
        gray = gray[:h_trim, :w_trim]

        # Prepare list for AC coefficients.
        ac_coeffs = []

        # Iterate over non-overlapping blocks.
        for y in range(0, h_trim, bs):
            for x in range(0, w_trim, bs):
                block = gray[y : y + bs, x : x + bs]
                # 2D DCT: first along rows, then columns.
                block_dct = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")

                # Flatten and remove DC coefficient at (0, 0).
                flat = block_dct.flatten()
                ac = flat[1:]  # all except the very first (DC)
                ac_coeffs.append(ac)

        if not ac_coeffs:
            return 0.0

        ac_coeffs = np.concatenate(ac_coeffs, axis=0)

        # Work with magnitudes to be sign-agnostic.
        ac_magnitudes = np.abs(ac_coeffs)

        if ac_magnitudes.size == 0:
            return 0.0

        # Build histogram of AC magnitudes.
        hist, bin_edges = np.histogram(ac_magnitudes, bins=bins, range=(0, ac_magnitudes.max() + 1e-6))

        # Identify "holes" in the histogram: bins with zero counts
        # amidst non-zero bins. Compute proportion of zero bins and
        # their relative depth as a heuristic tamper score.
        zero_bins = (hist == 0)

        # Exclude leading/trailing empty bins that may just represent tails.
        if zero_bins.any():
            first_nonzero = np.argmax(hist > 0)
            last_nonzero = len(hist) - 1 - np.argmax(hist[::-1] > 0)
            core_hist = hist[first_nonzero : last_nonzero + 1]
        else:
            core_hist = hist

        if core_hist.size == 0:
            return 0.0

        core_zero_mask = (core_hist == 0)
        zero_ratio = core_zero_mask.mean()

        # Depth measure: compare non-zero bins to their mean.
        nonzero_vals = core_hist[core_hist > 0]
        if nonzero_vals.size == 0:
            return 0.0

        mean_nonzero = nonzero_vals.mean()
        depth_score = float(np.clip(mean_nonzero / (mean_nonzero + 1.0), 0.0, 1.0))

        # Combine ratio of holes and depth into a single tamper score.
        dq_score = float(np.clip(zero_ratio * depth_score * 2.0, 0.0, 1.0))
        return dq_score

    # --- PRNU (Sensor Fingerprinting) core routine ---

    def run_prnu(
        self,
        h: float = 10.0,
        template_window_size: int = 7,
        search_window_size: int = 21,
    ) -> float:
        """
        Extract a PRNU-like noise residual and score how much it deviates from
        typical sensor-noise expectations.

        This is a *no-reference* PRNU heuristic (no camera fingerprint database):
        - Denoise the image to estimate the clean signal.
        - Subtract to obtain a high-frequency residual (noise pattern).
        - Score abnormal residual structure/energy (e.g., unusually strong,
          unusually weak, or spatially structured residuals).

        Parameters
        ----------
        h : float, optional
            Filter strength for `cv2.fastNlMeansDenoising` (default 10.0).
        template_window_size : int, optional
            Template window size for NLM (default 7).
        search_window_size : int, optional
            Search window size for NLM (default 21).

        Returns
        -------
        float
            Normalized tamper score in [0, 1]. Higher can indicate suspicious
            deviation from expected sensor-noise behavior.
        """
        if self.image is None:
            raise ValueError("Image not loaded for PRNU analysis.")

        gray_u8 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # NLM denoise (expects 8-bit single channel).
        denoised_u8 = cv2.fastNlMeansDenoising(
            gray_u8,
            None,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size,
        )

        gray = gray_u8.astype(np.float32)
        denoised = denoised_u8.astype(np.float32)

        # Noise residual (high-frequency pattern proxy).
        residual = gray - denoised
        residual -= float(residual.mean())

        # 1) Energy of residual relative to signal (too low => heavy smoothing / synthetic,
        #    too high => strong artifacts / inconsistent processing).
        signal_std = float(np.std(gray)) + 1e-6
        residual_std = float(np.std(residual))
        ratio = residual_std / signal_std  # typical small number

        # 2) Spatial structure: PRNU residual should be mostly weakly correlated.
        #    Large neighbor correlation suggests structured artifacts (e.g., edits).
        r = residual
        r_center = r[1:-1, 1:-1]
        r_right = r[1:-1, 2:]
        r_down = r[2:, 1:-1]
        denom = float(np.sqrt(np.mean(r_center**2) * np.mean(r_right**2)) + 1e-6)
        corr_x = float(np.mean(r_center * r_right) / denom)
        denom = float(np.sqrt(np.mean(r_center**2) * np.mean(r_down**2)) + 1e-6)
        corr_y = float(np.mean(r_center * r_down) / denom)
        corr = float((abs(corr_x) + abs(corr_y)) / 2.0)

        # Map features to [0, 1] via simple, tunable heuristics.
        # ratio_score: penalize both very low and very high residual energy.
        # "sweet spot" rough range chosen empirically; adjust per dataset.
        low, high = 0.015, 0.12
        if ratio < low:
            ratio_score = float(np.clip((low - ratio) / low, 0.0, 1.0))
        elif ratio > high:
            ratio_score = float(np.clip((ratio - high) / high, 0.0, 1.0))
        else:
            ratio_score = 0.0

        # corr_score: higher correlation => more suspicious.
        corr_score = float(np.clip((corr - 0.03) / 0.12, 0.0, 1.0))

        prnu_score = float(np.clip(0.55 * corr_score + 0.45 * ratio_score, 0.0, 1.0))
        return prnu_score

    # --- CFA (Bayer Grid Analysis) core routine ---

    def run_cfa(self, ksize: int = 3, blur_ksize: int = 7) -> float:
        """
        Detect CFA (Color Filter Array) inconsistencies using Laplacian-based
        local variance heuristics.

        Idea:
        - Compute Laplacian response per color channel (high-frequency emphasis).
        - Measure local variance of the Laplacian magnitude (via blur of squared).
        - In genuine camera images, demosaicing tends to produce correlated edge
          behavior across channels; strong local disagreement can be suspicious.

        Parameters
        ----------
        ksize : int, optional
            Laplacian kernel size (default 3).
        blur_ksize : int, optional
            Gaussian blur kernel for local variance smoothing (default 7).

        Returns
        -------
        float
            Normalized tamper score in [0, 1]. Higher indicates stronger local
            cross-channel inconsistencies in high-frequency content.
        """
        if self.image is None:
            raise ValueError("Image not loaded for CFA analysis.")

        if self.image.ndim != 3 or self.image.shape[2] < 3:
            return 0.0

        img = self.image[:, :, :3].astype(np.float32) / 255.0  # BGR
        b, g, r = cv2.split(img)

        def lap_var(channel: np.ndarray) -> np.ndarray:
            lap = cv2.Laplacian(channel, cv2.CV_32F, ksize=ksize)
            mag2 = lap * lap
            k = int(blur_ksize)
            if k % 2 == 0:
                k += 1
            local = cv2.GaussianBlur(mag2, (k, k), 0)
            return local

        vb = lap_var(b)
        vg = lap_var(g)
        vr = lap_var(r)

        # Cross-channel disagreement map (relative differences).
        eps = 1e-6
        mean_v = (vb + vg + vr) / 3.0
        disagreement = (np.abs(vb - mean_v) + np.abs(vg - mean_v) + np.abs(vr - mean_v)) / (mean_v + eps)

        # Aggregate: fraction of pixels above a robust threshold.
        # Use 95th percentile baseline to adapt to image content.
        p95 = float(np.percentile(disagreement, 95))
        thresh = max(1.5, p95)  # enforce minimum sensitivity floor
        frac = float((disagreement > thresh).mean())

        # Map to [0,1] (typical frac is small; amplify moderately).
        score = float(np.clip(frac * 6.0, 0.0, 1.0))
        return score

    # --- SVD (Copy-Move Detection) core routine ---

    def run_svd_copy_move(
        self,
        block_size: int = 16,
        step: int = 8,
        k: int = 8,
        feature_decimals: int = 3,
        min_shift: int = 12,
    ) -> float:
        """
        Copy-move forgery detection via SVD block features and lexicographical sorting.

        Steps:
        - Convert to grayscale float32.
        - Extract overlapping blocks (block_size, step).
        - For each block, compute SVD and keep top-k singular values.
        - Normalize and quantize features (rounding) to stabilize matching.
        - Lexicographically sort features and compare neighboring entries
          to find near-identical blocks.

        Parameters
        ----------
        block_size : int, optional
            Size of square blocks (default 16).
        step : int, optional
            Sliding step for overlapping blocks (default 8).
        k : int, optional
            Number of singular values to keep (default 8).
        feature_decimals : int, optional
            Rounding decimals for feature quantization (default 3).
        min_shift : int, optional
            Minimum spatial shift between matched blocks to avoid trivial neighbors (default 12).

        Returns
        -------
        float
            Normalized tamper score in [0, 1] based on the density of consistent block matches.
        """
        if self.image is None:
            raise ValueError("Image not loaded for SVD copy-move analysis.")

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        bs = int(block_size)
        st = int(step)
        if h < bs or w < bs:
            return 0.0

        feats = []
        positions = []

        for y in range(0, h - bs + 1, st):
            for x in range(0, w - bs + 1, st):
                block = gray[y : y + bs, x : x + bs]
                # SVD on block
                try:
                    s = np.linalg.svd(block, compute_uv=False)
                except np.linalg.LinAlgError:
                    continue

                s = s[:k]
                s_norm = s / (s[0] + 1e-6)
                s_q = np.round(s_norm, decimals=int(feature_decimals))
                feats.append(s_q)
                positions.append((y, x))

        if len(feats) < 10:
            return 0.0

        feats = np.vstack(feats)  # (N,k)
        positions = np.array(positions, dtype=np.int32)

        # Lexicographic sort by feature columns.
        keys = tuple(feats[:, i] for i in range(feats.shape[1] - 1, -1, -1))
        order = np.lexsort(keys)
        feats_s = feats[order]
        pos_s = positions[order]

        # Compare neighboring feature vectors after sorting.
        matches = 0
        considered = 0
        matched_blocks: List[Tuple[int, int, int, int]] = []
        for i in range(len(feats_s) - 1):
            f1 = feats_s[i]
            f2 = feats_s[i + 1]
            if np.allclose(f1, f2, atol=10 ** (-int(feature_decimals))):
                y1, x1 = pos_s[i]
                y2, x2 = pos_s[i + 1]
                if abs(y1 - y2) + abs(x1 - x2) >= int(min_shift):
                    matches += 1
                    matched_blocks.append((int(y1), int(x1), int(y2), int(x2)))
            considered += 1

        if considered == 0:
            return 0.0

        match_ratio = matches / considered

        # Cache matching block coordinates for visualization.
        self._last_svd_matches = matched_blocks

        # Map match ratio to [0,1]. Genuine images usually have very low ratio.
        score = float(np.clip(match_ratio * 25.0, 0.0, 1.0))
        return score

    # --- Metadata & Hex Audit core routine ---

    def run_metadata_hex_audit(self) -> float:
        """
        Perform metadata + raw-bytes (hex) audit to flag editing footprints.

        - Reads EXIF tags (when available) and searches for common editor names.
        - Scans raw file bytes for editor strings (ASCII and basic UTF-16LE).
        - Flags missing camera-specific EXIF tags that are commonly present in
          real camera photos (but often absent in screenshots/edited exports).

        Returns
        -------
        float
            Normalized tamper score in [0, 1].
        """
        if not self.image_path:
            raise ValueError("No image path provided for metadata audit.")

        footprints = ["photoshop", "gimp", "canva", "lightroom", "snapseed", "picsart", "pixlr"]
        footprint_hits = 0
        missing_tags_hits = 0

        # EXIF scan
        try:
            im = Image.open(self.image_path)
            exif = None
            if hasattr(im, "getexif"):
                exif = im.getexif()
            elif hasattr(im, "_getexif"):
                exif = im._getexif()

            exif_dict = {}
            if exif:
                for k, v in dict(exif).items():
                    tag = ExifTags.TAGS.get(k, str(k))
                    exif_dict[str(tag)] = v

            # Search EXIF values for footprints.
            combined = " ".join(str(v) for v in exif_dict.values()).lower()
            for fp in footprints:
                if fp in combined:
                    footprint_hits += 1

            # Missing camera-ish tags heuristic
            expected = [
                "Make",
                "Model",
                "DateTimeOriginal",
                "ExifVersion",
                "FNumber",
                "ExposureTime",
                "ISOSpeedRatings",
                "FocalLength",
            ]
            for t in expected:
                if t not in exif_dict or exif_dict.get(t) in (None, "", 0):
                    missing_tags_hits += 1

        except Exception:
            # If Pillow can't read EXIF (or file isn't an image), treat as suspicious.
            missing_tags_hits += 4

        # Raw bytes scan
        try:
            with open(self.image_path, "rb") as f:
                data = f.read()

            # ASCII-ish scan
            ascii_text = data.decode("latin-1", errors="ignore").lower()
            # Basic UTF-16LE scan (common in some metadata blocks)
            utf16_text = None
            try:
                utf16_text = data.decode("utf-16le", errors="ignore").lower()
            except Exception:
                utf16_text = ""

            for fp in footprints:
                pat = re.compile(re.escape(fp))
                if pat.search(ascii_text) or pat.search(utf16_text):
                    footprint_hits += 1

        except Exception:
            # Can't read bytes => suspicious but not definitive.
            footprint_hits += 1

        # Score composition
        footprint_score = float(np.clip(footprint_hits / 3.0, 0.0, 1.0))
        missing_score = float(np.clip(missing_tags_hits / 8.0, 0.0, 1.0))

        score = float(np.clip(0.65 * footprint_score + 0.35 * missing_score, 0.0, 1.0))
        return score
    # --- Six Pillars of Detection (method stubs) ---

    def detect_ela(self) -> float:
        """
        Error Level Analysis (ELA) pillar.

        Returns
        -------
        float
            Confidence score in the range [0, 1],
            where higher values can represent higher likelihood of tampering.
        """
        return self.run_ela()

    def detect_double_quantization(self) -> float:
        """
        Double Quantization (JPEG) pillar.

        Returns
        -------
        float
            Confidence score in the range [0, 1].
        """
        return self.run_double_quantization()

    def detect_prnu(self) -> float:
        """
        Photo Response Non-Uniformity (PRNU) pillar.

        Returns
        -------
        float
            Confidence score in the range [0, 1].
        """
        return self.run_prnu()

    def detect_cfa(self) -> float:
        """
        Color Filter Array (CFA) inconsistencies pillar.

        Returns
        -------
        float
            Confidence score in the range [0, 1].
        """
        return self.run_cfa()

    def detect_svd(self) -> float:
        """
        Singular Value Decomposition (SVD) based analysis pillar.

        Returns
        -------
        float
            Confidence score in the range [0, 1].
        """
        return self.run_svd_copy_move()

    def audit_metadata(self) -> float:
        """
        Metadata Audit pillar (EXIF and related metadata).

        Returns
        -------
        float
            Confidence score in the range [0, 1].
        """
        return self.run_metadata_hex_audit()

    # --- Aggregation / Truth Score ---

    def generate_truth_score(self, weights: Dict[str, float] | None = None) -> float:
        """
        Aggregate the six pillars into a single probability of tampering
        via weighted average.

        Parameters
        ----------
        weights : dict[str, float], optional
            Mapping pillar name -> weight. If not provided, all pillars
            are weighted equally.

            Expected keys:
            - "ela"
            - "dq"          (double quantization)
            - "prnu"
            - "cfa"
            - "svd"
            - "metadata"

        Returns
        -------
        float
            Probability of tampering in the range [0, 1].
        """
        # Default to equal weights if none are provided.
        if weights is None:
            weights = {
                "ela": 1.0,
                "dq": 1.0,
                "prnu": 1.0,
                "cfa": 1.0,
                "svd": 1.0,
                "metadata": 1.0,
            }

        # Collect scores from each pillar.
        scores = {
            "ela": self.detect_ela(),
            "dq": self.detect_double_quantization(),
            "prnu": self.detect_prnu(),
            "cfa": self.detect_cfa(),
            "svd": self.detect_svd(),
            "metadata": self.audit_metadata(),
        }

        # Ensure we are working with NumPy arrays.
        w = np.array([weights[k] for k in weights], dtype=float)
        s = np.array([scores[k] for k in weights], dtype=float)

        # Avoid division by zero if all weights are zero.
        weight_sum = w.sum()
        if weight_sum == 0:
            raise ValueError("Sum of weights must be non-zero.")

        probability_of_tampering = float(np.average(s, weights=w))
        # Clamp to [0, 1] for safety.
        probability_of_tampering = float(np.clip(probability_of_tampering, 0.0, 1.0))

        return probability_of_tampering

    # --- Visual Tamper Maps ---

    def generate_visual_report(self, results_dir: str = "results") -> Dict[str, object]:
        """
        Generate localized tamper maps for visualization.

        - ELA pillar: saves a grayscale heatmap where brighter pixels
          indicate higher probability of tampering.
        - SVD (Copy-Move) pillar: returns coordinates of matching blocks
          and saves a PNG map with the matched blocks highlighted.

        Parameters
        ----------
        results_dir : str, optional
            Directory where PNG files will be written (default "results").

        Returns
        -------
        dict
            Dictionary with:
              - "ela_heatmap_path": str
              - "svd_tamper_map_path": str
              - "svd_matches": List[Tuple[int, int, int, int]]
                (y1, x1, y2, x2) for matching block pairs.
        """
        if self.image is None:
            raise ValueError("Image not loaded for visual report generation.")

        os.makedirs(results_dir, exist_ok=True)

        # Derive a clean base name for output files.
        base_name = os.path.splitext(os.path.basename(self.image_path or "image"))[0]

        # --- ELA tamper map ---
        _ = self.run_ela()
        if self._last_ela_map is None:
            ela_map = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        else:
            ela_map = self._last_ela_map

        ela_out_path = os.path.join(results_dir, f"{base_name}_ELA_tamper_map.png")
        # Ensure it is single-channel grayscale PNG.
        cv2.imwrite(ela_out_path, ela_map)

        # --- SVD copy-move map ---
        _ = self.run_svd_copy_move()
        matches = getattr(self, "_last_svd_matches", []) or []

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        svd_map = np.zeros_like(gray, dtype=np.uint8)

        # Highlight both blocks in each matching pair.
        for (y1, x1, y2, x2) in matches:
            # Use a fixed block size heuristic tied to map resolution:
            # approximate local region around the block coordinates.
            block_radius = 8
            for (y, x) in [(y1, x1), (y2, x2)]:
                y0 = max(0, y)
                x0 = max(0, x)
                y1b = min(svd_map.shape[0], y + block_radius)
                x1b = min(svd_map.shape[1], x + block_radius)
                svd_map[y0:y1b, x0:x1b] = 255

        svd_out_path = os.path.join(results_dir, f"{base_name}_SVD_tamper_map.png")
        cv2.imwrite(svd_out_path, svd_map)

        return {
            "ela_heatmap_path": ela_out_path,
            "svd_tamper_map_path": svd_out_path,
            "svd_matches": matches,
        }

