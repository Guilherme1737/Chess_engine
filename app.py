"""Local screen-capture chess coach with piece recognition, figurine notation, and UCI analysis."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from coach_core import (
    AppSettings,
    BoardEditModel,
    HistoryEntry,
    PositionHistory,
    TrainingAttempt,
    TrainingSession,
    ValidationReport,
    build_session_report,
    evaluate_guess,
    explain_move,
    find_move_between,
    format_analysis_for_clipboard,
    is_suspicious,
    load_settings,
    position_key,
    sanitize_fen,
    save_settings,
    validate_fen,
)

DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEFAULT_START_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration_data.npz"
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"
ENGINES_DIR = Path(__file__).resolve().parent / "engines"
PIECE_TYPES = ("P", "N", "B", "R", "Q", "K")


def find_default_engine() -> str:
    """Finds a bundled Stockfish/UCI binary in ./engines or on the system PATH."""
    if ENGINES_DIR.is_dir():
        for ext in ("*.exe", "stockfish*"):
            for candidate in sorted(ENGINES_DIR.rglob(ext)):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)
    for name in ("stockfish", "opental"):
        found = shutil.which(name)
        if found:
            return found
    return ""

FIGURINE_MAP_WHITE = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙"}
FIGURINE_MAP_BLACK = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}
PT_PIECE_NAMES = {
    "P": "Peao",
    "N": "Cavalo",
    "B": "Bispo",
    "R": "Torre",
    "Q": "Dama",
    "K": "Rei",
}


@dataclass(frozen=True)
class BoardRegion:
    x: int
    y: int
    width: int
    height: int


@dataclass
class RecognitionResult:
    fen: str
    board_fen: str
    confidence: float
    uncertain_squares: List[str] = field(default_factory=list)
    calibrated: bool = False
    detected_orientation: Optional[str] = None
    inferred_turn: Optional[str] = None
    last_move_uci: Optional[str] = None


@dataclass
class EngineLine:
    multipv: int
    uci_move: str
    san_move: str
    figurine_move: str
    piece_symbol: str  # 'P','N','B','R','Q','K' (uppercase for white, lowercase for black)
    score_white_cp: Optional[int]  # Centipawns from White's perspective (+ is White advantage)
    mate_white: Optional[int]  # Mate in N from White's perspective (+M is White mating, -M is Black mating)
    eval_text: str  # Formatted like "+0.35", "-1.20", "+M2"
    pv_figurine: str  # Formatted PV like "1. ♘f3 ♞c6 2. ♗b5"
    pv_uci: List[str] = field(default_factory=list)  # Raw PV in UCI
    pv_san: str = ""  # Plain SAN PV like "1. Nf3 Nc6 2. Bb5" (for export/PGN)
    depth: Optional[int] = None  # Search depth reached for this line (if known)


def san_to_figurine(san: str, is_white: bool) -> str:
    """Converts standard SAN (e.g. 'Nf3', 'Qh5+') into Figurine Algebraic Notation ('♘f3', '♕h5+')."""
    if not san:
        return san
    fmap = FIGURINE_MAP_WHITE if is_white else FIGURINE_MAP_BLACK
    if san[0] in ("K", "Q", "R", "B", "N"):
        return fmap[san[0]] + san[1:]
    for letter in ("Q", "R", "B", "N"):
        target = f"={letter}"
        if target in san:
            san = san.replace(target, f"={fmap[letter]}")
    return san


def format_eval_badge(score_white_cp: Optional[int], mate_white: Optional[int]) -> str:
    """Formats evaluation in Lichess/Chess.com style (+0.45, -1.20, +M3, -M1)."""
    if mate_white is not None:
        sign = "+" if mate_white > 0 else "-"
        return f"{sign}M{abs(mate_white)}"
    if score_white_cp is None:
        return "0.00"
    pawns = score_white_cp / 100.0
    sign = "+" if pawns > 0 else ""
    return f"{sign}{pawns:.2f}"


def infer_turn_and_last_move(
    prev_board_fen: Optional[str],
    curr_board_fen: str,
    fallback_turn: str = "w",
    visual_turn: Optional[str] = None,
    visual_last_move: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Automatically infers whose turn it is ('w' or 'b') and the last move played (UCI)
    by combining:
    1. Initial position rule (always White's turn);
    2. King-in-check rule (if a king is attacked, it MUST be that king's turn);
    3. Visual highlight detection on the board (empty highlighted origin + occupied highlighted destination);
    4. Exact legal move transition from the previous captured board state (or from the start of a new game).
    """
    import chess

    # 1. Initial position is always White to move
    if curr_board_fen == DEFAULT_START_BOARD_FEN:
        return "w", None

    curr_base = chess.BaseBoard(curr_board_fen)

    # 2. If a king is currently in check, it MUST be that king's turn to move
    white_king_sq = curr_base.king(chess.WHITE)
    black_king_sq = curr_base.king(chess.BLACK)
    w_attacked = white_king_sq is not None and curr_base.is_attacked_by(chess.BLACK, white_king_sq)
    b_attacked = black_king_sq is not None and curr_base.is_attacked_by(chess.WHITE, black_king_sq)
    if w_attacked and not b_attacked:
        inferred_by_check = "w"
    elif b_attacked and not w_attacked:
        inferred_by_check = "b"
    else:
        inferred_by_check = None

    # 3. If visual highlight detection found a clear last move on the board (from_sq empty -> to_sq occupied)
    if visual_turn in ("w", "b") and visual_last_move:
        turn_out = inferred_by_check if inferred_by_check is not None else visual_turn
        return turn_out, visual_last_move

    # 4. Compare against previous board position if available and adjacent (<= 4 changed squares)
    if prev_board_fen and prev_board_fen != curr_board_fen:
        found = find_move_between(f"{prev_board_fen} w KQkq - 0 1", f"{curr_board_fen} w - - 0 1")
        if found is not None:
            mv, before_board = found
            next_turn_code = "b" if before_board.turn == chess.WHITE else "w"
            return (inferred_by_check or next_turn_code), mv.uci()

        try:
            prev_base = chess.BaseBoard(prev_board_fen)
            changed_squares = sum(
                1 for sq in chess.SQUARES if prev_base.piece_at(sq) != curr_base.piece_at(sq)
            )
            # Only infer from vacated squares if the transition is a single ply (at most 4 squares changed, e.g. castling)
            if 1 <= changed_squares <= 4:
                vacated_white: List[int] = []
                vacated_black: List[int] = []
                arrived_white: List[int] = []
                arrived_black: List[int] = []
                for sq in chess.SQUARES:
                    p_prev = prev_base.piece_at(sq)
                    p_curr = curr_base.piece_at(sq)
                    if p_prev is not None and p_curr is None:
                        if p_prev.color == chess.WHITE:
                            vacated_white.append(sq)
                        else:
                            vacated_black.append(sq)
                    elif p_curr is not None and p_curr != p_prev:
                        if p_curr.color == chess.WHITE:
                            arrived_white.append(sq)
                        else:
                            arrived_black.append(sq)

                if vacated_white and not vacated_black:
                    last_uci = None
                    if len(vacated_white) == 1 and len(arrived_white) == 1:
                        last_uci = chess.square_name(vacated_white[0]) + chess.square_name(arrived_white[0])
                    return (inferred_by_check or "b"), last_uci
                if vacated_black and not vacated_white:
                    last_uci = None
                    if len(vacated_black) == 1 and len(arrived_black) == 1:
                        last_uci = chess.square_name(vacated_black[0]) + chess.square_name(arrived_black[0])
                    return (inferred_by_check or "w"), last_uci
        except Exception:
            pass

    # 5. Check if the board is 1 move after the initial starting position (e.g. White just played move 1 in a new game)
    from_start = find_move_between(DEFAULT_START_FEN, f"{curr_board_fen} w KQkq - 0 1")
    if from_start is not None:
        mv, _ = from_start
        return (inferred_by_check or "b"), mv.uci()

    if inferred_by_check is not None:
        return inferred_by_check, visual_last_move
    if visual_turn in ("w", "b"):
        return visual_turn, visual_last_move

    return fallback_turn, visual_last_move


def build_engine_line(
    fen: str,
    multipv: int,
    pv_uci: List[str],
    score_turn_cp: Optional[int] = None,
    mate_turn: Optional[int] = None,
    depth: Optional[int] = None,
) -> EngineLine:
    """Constructs a rich EngineLine with SAN, Figurine notation, and White-perspective score."""
    import chess

    board = chess.Board(fen)
    if not pv_uci:
        raise ValueError("PV vazia")

    first_uci = pv_uci[0]
    first_move = chess.Move.from_uci(first_uci)
    if first_move not in board.legal_moves:
        raise ValueError(f"Lance ilegal {first_uci} para a FEN {fen}")

    is_white_turn = board.turn == chess.WHITE
    moved_piece = board.piece_at(first_move.from_square)
    piece_symbol = moved_piece.symbol() if moved_piece else ("P" if is_white_turn else "p")
    san_move = board.san(first_move)
    figurine_move = san_to_figurine(san_move, is_white_turn)

    sign = 1 if is_white_turn else -1
    score_white_cp = score_turn_cp * sign if score_turn_cp is not None else None
    mate_white = mate_turn * sign if mate_turn is not None else None
    eval_text = format_eval_badge(score_white_cp, mate_white)

    pv_parts: List[str] = []
    san_parts: List[str] = []
    legal_pv: List[str] = []
    sim_board = board.copy()
    for idx, uci_str in enumerate(pv_uci[:12]):
        try:
            mv = chess.Move.from_uci(uci_str)
            if mv not in sim_board.legal_moves:
                break
            w_turn = sim_board.turn == chess.WHITE
            mv_san = sim_board.san(mv)
            mv_fig = san_to_figurine(mv_san, w_turn)
            if w_turn:
                prefix = f"{sim_board.fullmove_number}. "
            elif idx == 0:
                prefix = f"{sim_board.fullmove_number}... "
            else:
                prefix = ""
            if idx < 6:
                pv_parts.append(prefix + mv_fig)
            san_parts.append(prefix + mv_san)
            legal_pv.append(uci_str)
            sim_board.push(mv)
        except ValueError:
            break

    return EngineLine(
        multipv=multipv,
        uci_move=first_uci,
        san_move=san_move,
        figurine_move=figurine_move,
        piece_symbol=piece_symbol,
        score_white_cp=score_white_cp,
        mate_white=mate_white,
        eval_text=eval_text,
        pv_figurine=" ".join(pv_parts),
        pv_uci=legal_pv,
        pv_san=" ".join(san_parts),
        depth=depth,
    )


def parse_user_move_to_uci(fen: str, user_input: str) -> str:
    """Parses user move input in UCI ('g1f3'), English SAN ('Nf3'), or Portuguese SAN ('Cf3', 'Bc4', 'Td1')."""
    import chess

    raw = user_input.strip()
    if not raw:
        raise ValueError("Lance vazio")
    board = chess.Board(fen)

    for k, fig in list(FIGURINE_MAP_WHITE.items()) + list(FIGURINE_MAP_BLACK.items()):
        if raw.startswith(fig):
            raw = ("" if k == "P" else k) + raw[len(fig) :]
        if f"={fig}" in raw:
            raw = raw.replace(f"={fig}", f"={k}")

    pt_to_en = {"C": "N", "B": "B", "T": "R", "D": "Q", "R": "K"}
    for pt_ch, en_ch in pt_to_en.items():
        if f"={pt_ch}" in raw:
            raw = raw.replace(f"={pt_ch}", f"={en_ch}")

    try:
        return board.parse_san(raw).uci()
    except ValueError:
        pass

    if raw[0] in pt_to_en:
        en_san = pt_to_en[raw[0]] + raw[1:]
        try:
            return board.parse_san(en_san).uci()
        except ValueError:
            pass

    try:
        return board.parse_uci(raw.lower()).uci()
    except ValueError as exc:
        raise ValueError(f"Lance invalido ou ilegal na posicao: '{user_input}'") from exc


def parse_region(value: str) -> BoardRegion:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("A regiao deve usar o formato x,y,w,h") from exc
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError("A regiao deve usar o formato x,y,w,h com largura e altura positivas")
    return BoardRegion(*parts)


class ScreenCapture:
    def capture(self):
        try:
            import mss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Instale as dependencias com: python -m pip install -r requirements.txt") from exc

        with mss.mss() as monitor:
            image = np.array(monitor.grab(monitor.monitors[0]))
        return image[:, :, :3]

    def crop_region(self, image, region: BoardRegion):
        height, width = image.shape[:2]
        x1 = max(0, min(region.x, width - 1))
        y1 = max(0, min(region.y, height - 1))
        x2 = min(width, x1 + region.width)
        y2 = min(height, y1 + region.height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("A regiao esta fora da tela capturada")
        return image[y1:y2, x1:x2].copy()

    def save_region(self, image, region: BoardRegion, path: str) -> None:
        import cv2

        cropped = self.crop_region(image, region)
        cv2.imwrite(path, cropped)


class BoardDetector:
    """Finds likely square contours, independent of board theme and orientation."""

    def detect(self, image) -> BoardRegion:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width = gray.shape
        candidates = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            x, y, width, height = cv2.boundingRect(polygon)
            area = width * height
            ratio = width / max(height, 1)
            if (
                len(polygon) == 4
                and 0.88 <= ratio <= 1.14
                and image_width * image_height * 0.03 < area < image_width * image_height * 0.90
            ):
                candidates.append((area, BoardRegion(x, y, width, height)))
        if not candidates:
            raise RuntimeError("Nenhum tabuleiro foi detectado; informe x,y,w,h ou selecione com o mouse")
        return max(candidates, key=lambda candidate: candidate[0])[1]


class PieceRecognizer:
    """Recognizes chess pieces on an 8x8 cropped board image using OpenCV templates."""

    BOARD_SIZE = 512
    SQUARE_SIZE = 64
    MARGIN = 6  # Inner patch is 52x52

    def __init__(self, calibration_path: Optional[Path] = DEFAULT_CALIBRATION_PATH):
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.bg_light: Optional[object] = None
        self.bg_dark: Optional[object] = None
        self.templates: Dict[str, Dict[str, object]] = {}
        self.template_banks: List[Dict[str, Dict[str, object]]] = []
        # Extra template variants learned from manual corrections in the board editor (per piece symbol)
        self.correction_samples: Dict[str, List[Dict[str, object]]] = {}
        self.calibrated = False
        self._synthetic_masks = self._build_synthetic_masks()
        if self.calibration_path and self.calibration_path.is_file():
            try:
                self.load_calibration(self.calibration_path)
            except Exception:
                pass

    @staticmethod
    def _build_synthetic_masks() -> Dict[str, object]:
        """Builds 52x52 synthetic silhouette templates for fallback recognition and piece icons."""
        import cv2
        import numpy as np

        size = 52
        masks: Dict[str, object] = {}

        p = np.zeros((size, size), dtype=np.float32)
        cv2.circle(p, (26, 17), 7, 1.0, -1)
        cv2.fillPoly(p, [np.array([[20, 24], [32, 24], [35, 39], [17, 39]])], 1.0)
        cv2.rectangle(p, (13, 39), (39, 45), 1.0, -1)
        masks["P"] = p

        r = np.zeros((size, size), dtype=np.float32)
        cv2.rectangle(r, (14, 11), (38, 18), 1.0, -1)
        cv2.rectangle(r, (18, 18), (34, 38), 1.0, -1)
        cv2.rectangle(r, (12, 38), (40, 45), 1.0, -1)
        masks["R"] = r

        n = np.zeros((size, size), dtype=np.float32)
        pts_n = np.array([[16, 44], [38, 44], [36, 26], [33, 12], [22, 10], [13, 21], [21, 25], [16, 36]])
        cv2.fillPoly(n, [pts_n], 1.0)
        masks["N"] = n

        b = np.zeros((size, size), dtype=np.float32)
        cv2.circle(b, (26, 10), 3, 1.0, -1)
        cv2.ellipse(b, (26, 24), (9, 12), 0, 0, 360, 1.0, -1)
        cv2.rectangle(b, (13, 39), (39, 45), 1.0, -1)
        masks["B"] = b

        q = np.zeros((size, size), dtype=np.float32)
        pts_q = np.array([
            [14, 40], [10, 16], [19, 26], [26, 12], [33, 26], [42, 16], [38, 40]
        ])
        cv2.fillPoly(q, [pts_q], 1.0)
        cv2.rectangle(q, (12, 39), (40, 45), 1.0, -1)
        masks["Q"] = q

        k = np.zeros((size, size), dtype=np.float32)
        cv2.rectangle(k, (24, 6), (28, 15), 1.0, -1)
        cv2.rectangle(k, (21, 8), (31, 12), 1.0, -1)
        cv2.ellipse(k, (26, 27), (12, 13), 0, 0, 360, 1.0, -1)
        cv2.rectangle(k, (12, 39), (40, 45), 1.0, -1)
        masks["K"] = k

        return masks

    def _normalize_board(self, board_image):
        import cv2

        if board_image is None or board_image.size == 0:
            raise ValueError("Imagem do tabuleiro vazia")
        return cv2.resize(board_image, (self.BOARD_SIZE, self.BOARD_SIZE), interpolation=cv2.INTER_AREA)

    def _extract_square_patch(self, norm_board, row: int, col: int):
        s = self.SQUARE_SIZE
        m = self.MARGIN
        y1 = row * s + m
        y2 = (row + 1) * s - m
        x1 = col * s + m
        x2 = (col + 1) * s - m
        return norm_board[y1:y2, x1:x2]

    def _estimate_backgrounds(self, norm_board):
        import numpy as np

        light_borders = []
        dark_borders = []
        for r in range(8):
            for c in range(8):
                patch = self._extract_square_patch(norm_board, r, c).astype(np.float32)
                top = patch[:4, :, :].reshape(-1, 3)
                bottom = patch[-4:, :, :].reshape(-1, 3)
                left = patch[:, :4, :].reshape(-1, 3)
                right = patch[:, -4:, :].reshape(-1, 3)
                border = np.vstack([top, bottom, left, right])
                if (r + c) % 2 == 0:
                    light_borders.append(np.median(border, axis=0))
                else:
                    dark_borders.append(np.median(border, axis=0))
        bg_light = np.median(np.array(light_borders), axis=0).astype(np.float32)
        bg_dark = np.median(np.array(dark_borders), axis=0).astype(np.float32)
        return bg_light, bg_dark

    def _extract_features(self, patch, normal_bg_color) -> Dict[str, object]:
        """
        Extracts piece features using THIS square's own border background color,
        making recognition 100% immune to yellow/green/blue last-move square highlights!
        """
        import cv2
        import numpy as np

        patch_f = patch.astype(np.float32)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)

        # Estimate THIS square's own background from low-gradient pixels on its outer 4px ring
        border_mask = np.zeros(gray.shape, dtype=bool)
        border_mask[:4, :] = True
        border_mask[-4:, :] = True
        border_mask[:, :4] = True
        border_mask[:, -4:] = True
        flat_border = border_mask & (edges < 35.0)
        if np.count_nonzero(flat_border) >= 16:
            square_bg = np.median(patch_f[flat_border], axis=0)
        else:
            square_bg = np.median(patch_f[border_mask], axis=0)

        # How much this square's background differs from normal unhighlighted squares (detects last-move highlights)
        highlight_shift = float(np.linalg.norm(square_bg - normal_bg_color))

        # Pixel distance from THIS square's own background (eliminates yellow/green highlight background!)
        diff = np.linalg.norm(patch_f - square_bg[None, None, :], axis=2)

        fg_raw = (diff > 20.0).astype(np.uint8)
        contours, _ = cv2.findContours(fg_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fg_filled = np.zeros_like(fg_raw)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 8:
                cv2.drawContours(fg_filled, [cnt], -1, 1, thickness=cv2.FILLED)
        fg_mask = fg_filled.astype(np.float32)

        # Evaluate occupancy in the central 38x38 zone (ignores corner rank/file labels 'a-h', '1-8')
        center_fg = float(np.mean(fg_mask[7:45, 7:45]))
        center_diff = float(np.mean(diff[7:45, 7:45]))
        center_edge = float(np.mean(edges[7:45, 7:45]))
        center_std = float(np.std(gray[7:45, 7:45]))
        is_occupied = bool(
            center_fg > 0.11
            and center_std > 11.0
            and (center_diff > 11.0 or center_edge > 20.0)
        )

        interior = cv2.erode(fg_filled, np.ones((3, 3), np.uint8), iterations=1)
        if np.any(fg_filled > 0):
            body_luma = float(np.mean(gray[fg_filled > 0]))
        elif np.any(interior > 0):
            body_luma = float(np.mean(gray[interior > 0]))
        else:
            body_luma = 128.0

        neutral_gray = np.where(fg_mask > 0.5, gray, 128.0).astype(np.float32) / 255.0
        norm_edges = edges / (np.max(edges) + 1e-5)

        return {
            "diff_mean": center_diff,
            "edge_mean": center_edge,
            "fg_ratio": center_fg,
            "is_occupied": is_occupied,
            "highlight_shift": highlight_shift,
            "fg_mask": fg_mask,
            "neutral_gray": neutral_gray,
            "edges": norm_edges,
            "gray": gray,
            "body_luma": body_luma,
        }

    def detect_orientation(self, board_image) -> str:
        """
        Automatically detects board orientation ('white' or 'black' at bottom)
        by comparing the piece interior luminance in the bottom ranks vs top ranks.
        """
        import numpy as np

        norm_board = self._normalize_board(board_image)
        est_light, est_dark = self._estimate_backgrounds(norm_board)

        top_lumas: List[float] = []
        bottom_lumas: List[float] = []
        for r in range(8):
            for c in range(8):
                bg = est_light if (r + c) % 2 == 0 else est_dark
                patch = self._extract_square_patch(norm_board, r, c)
                feats = self._extract_features(patch, bg)
                if not feats["is_occupied"]:
                    continue
                if r <= 2:
                    top_lumas.append(feats["body_luma"])
                elif r >= 5:
                    bottom_lumas.append(feats["body_luma"])

        if top_lumas and bottom_lumas:
            return "white" if float(np.mean(bottom_lumas)) >= float(np.mean(top_lumas)) else "black"
        return "white"

    def calibrate(self, board_image, fen: str = DEFAULT_START_FEN, orientation: str = "white") -> int:
        """Extracts and saves piece templates from a board image with a known FEN."""
        import chess
        import numpy as np

        board = chess.Board(fen)
        norm_board = self._normalize_board(board_image)
        est_light, est_dark = self._estimate_backgrounds(norm_board)

        empty_light = []
        empty_dark = []
        grouped: Dict[str, List[Dict[str, object]]] = {}

        for r in range(8):
            for c in range(8):
                file_idx = c if orientation == "white" else 7 - c
                rank_idx = 7 - r if orientation == "white" else r
                sq = chess.square(file_idx, rank_idx)
                piece = board.piece_at(sq)
                bg = est_light if (r + c) % 2 == 0 else est_dark
                patch = self._extract_square_patch(norm_board, r, c)
                feats = self._extract_features(patch, bg)

                if piece is None and not feats["is_occupied"]:
                    if (r + c) % 2 == 0:
                        empty_light.append(np.median(patch.astype(np.float32).reshape(-1, 3), axis=0))
                    else:
                        empty_dark.append(np.median(patch.astype(np.float32).reshape(-1, 3), axis=0))
                elif piece is not None and feats["is_occupied"]:
                    symbol = piece.symbol()
                    grouped.setdefault(symbol, []).append(feats)

        self.bg_light = np.median(empty_light, axis=0).astype(np.float32) if empty_light else est_light
        self.bg_dark = np.median(empty_dark, axis=0).astype(np.float32) if empty_dark else est_dark

        if not grouped:
            raise ValueError("Nenhuma peca encontrada na FEN informada para calibracao")

        new_bank: Dict[str, Dict[str, object]] = {}
        for symbol, items in grouped.items():
            new_bank[symbol] = {
                "fg_mask": np.mean([it["fg_mask"] for it in items], axis=0).astype(np.float32),
                "neutral_gray": np.mean([it["neutral_gray"] for it in items], axis=0).astype(np.float32),
                "edges": np.mean([it["edges"] for it in items], axis=0).astype(np.float32),
            }

        self.templates = new_bank
        self.template_banks.append(new_bank)
        if len(self.template_banks) > 5:
            self.template_banks = self.template_banks[-5:]
        self.calibrated = True
        if self.calibration_path:
            self.save_calibration(self.calibration_path)
        return len(self.templates)

    MAX_CORRECTIONS_PER_PIECE = 8

    @property
    def theme_count(self) -> int:
        return len(self.template_banks) if self.template_banks else (1 if self.templates else 0)

    @property
    def correction_count(self) -> int:
        return sum(len(v) for v in self.correction_samples.values())

    def learn_from_correction(
        self,
        board_image,
        corrected_fen: str,
        orientation: str = "white",
        squares: Optional[List[str]] = None,
    ) -> int:
        """
        Learns extra template variants from a manually corrected position (board editor).
        Only the given squares (or all occupied squares) are learned. Returns how many samples were added.
        """
        import chess

        board = chess.BaseBoard(corrected_fen.split()[0])
        norm_board = self._normalize_board(board_image)
        est_light, est_dark = self._estimate_backgrounds(norm_board)
        wanted = set(squares) if squares is not None else None
        learned = 0
        for r in range(8):
            for c in range(8):
                file_idx = c if orientation == "white" else 7 - c
                rank_idx = 7 - r if orientation == "white" else r
                sq = chess.square(file_idx, rank_idx)
                if wanted is not None and chess.square_name(sq) not in wanted:
                    continue
                piece = board.piece_at(sq)
                if piece is None:
                    continue
                bg = est_light if (r + c) % 2 == 0 else est_dark
                feats = self._extract_features(self._extract_square_patch(norm_board, r, c), bg)
                if not feats["is_occupied"]:
                    continue
                samples = self.correction_samples.setdefault(piece.symbol(), [])
                samples.append({
                    "fg_mask": feats["fg_mask"],
                    "neutral_gray": feats["neutral_gray"],
                    "edges": feats["edges"],
                })
                del samples[: -self.MAX_CORRECTIONS_PER_PIECE]
                learned += 1
        if learned and self.calibration_path:
            self.save_calibration(self.calibration_path)
        return learned

    def clear_themes(self) -> None:
        """Removes all calibrated themes and learned corrections (falls back to synthetic silhouettes)."""
        self.templates = {}
        self.template_banks = []
        self.correction_samples = {}
        self.bg_light = None
        self.bg_dark = None
        self.calibrated = False
        if self.calibration_path and self.calibration_path.is_file():
            try:
                self.calibration_path.unlink()
            except OSError:
                pass

    def save_calibration(self, path: Path) -> None:
        import numpy as np

        banks = self.template_banks if self.template_banks else ([self.templates] if self.templates else [])
        if not banks and not self.correction_samples:
            return
        arrays: Dict[str, object] = {"bank_count": np.array(len(banks), dtype=np.int32)}
        if self.bg_light is not None and self.bg_dark is not None:
            arrays["bg_light"] = self.bg_light
            arrays["bg_dark"] = self.bg_dark
        if banks:
            latest = banks[-1]
            arrays["symbols"] = np.array(list(latest.keys()))
            for sym, data in latest.items():
                arrays[f"{sym}_fg_mask"] = data["fg_mask"]
                arrays[f"{sym}_neutral_gray"] = data["neutral_gray"]
                arrays[f"{sym}_edges"] = data["edges"]
        for b_idx, bank in enumerate(banks):
            arrays[f"b{b_idx}_symbols"] = np.array(list(bank.keys()))
            for sym, data in bank.items():
                arrays[f"b{b_idx}_{sym}_fg_mask"] = data["fg_mask"]
                arrays[f"b{b_idx}_{sym}_neutral_gray"] = data["neutral_gray"]
                arrays[f"b{b_idx}_{sym}_edges"] = data["edges"]
        corr_syms: List[str] = []
        for sym, samples in self.correction_samples.items():
            for sample in samples:
                i = len(corr_syms)
                corr_syms.append(sym)
                arrays[f"c{i}_fg_mask"] = sample["fg_mask"]
                arrays[f"c{i}_neutral_gray"] = sample["neutral_gray"]
                arrays[f"c{i}_edges"] = sample["edges"]
        if corr_syms:
            arrays["c_symbols"] = np.array(corr_syms)
        np.savez_compressed(path, **arrays)

    def load_calibration(self, path: Path) -> None:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            self.bg_light = data["bg_light"].astype(np.float32) if "bg_light" in data else None
            self.bg_dark = data["bg_dark"].astype(np.float32) if "bg_dark" in data else None
            self.template_banks = []
            self.correction_samples = {}
            if "bank_count" in data:
                b_count = int(data["bank_count"])
                for b_idx in range(b_count):
                    b_syms = [str(s) for s in data[f"b{b_idx}_symbols"]]
                    bank: Dict[str, Dict[str, object]] = {}
                    for sym in b_syms:
                        bank[sym] = {
                            "fg_mask": data[f"b{b_idx}_{sym}_fg_mask"].astype(np.float32),
                            "neutral_gray": data[f"b{b_idx}_{sym}_neutral_gray"].astype(np.float32),
                            "edges": data[f"b{b_idx}_{sym}_edges"].astype(np.float32),
                        }
                    if bank:
                        self.template_banks.append(bank)
            if not self.template_banks and "symbols" in data:
                symbols = [str(s) for s in data["symbols"]]
                bank = {}
                for sym in symbols:
                    bank[sym] = {
                        "fg_mask": data[f"{sym}_fg_mask"].astype(np.float32),
                        "neutral_gray": data[f"{sym}_neutral_gray"].astype(np.float32),
                        "edges": data[f"{sym}_edges"].astype(np.float32),
                    }
                if bank:
                    self.template_banks.append(bank)
            if "c_symbols" in data:
                for i, sym in enumerate(str(s) for s in data["c_symbols"]):
                    self.correction_samples.setdefault(sym, []).append({
                        "fg_mask": data[f"c{i}_fg_mask"].astype(np.float32),
                        "neutral_gray": data[f"c{i}_neutral_gray"].astype(np.float32),
                        "edges": data[f"c{i}_edges"].astype(np.float32),
                    })
            self.templates = self.template_banks[-1] if self.template_banks else {}
            self.calibrated = bool(self.templates)

    @staticmethod
    def _similarity(a, b) -> float:
        import cv2
        import numpy as np

        margin = 2
        tpl = b[margin:-margin, margin:-margin].astype(np.float32)
        src = a.astype(np.float32)
        if np.std(src) < 1e-5 or np.std(tpl) < 1e-5:
            return 0.0
        res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
        return float(np.max(res))

    @staticmethod
    def _mask_iou(a, b) -> float:
        import numpy as np

        inter = float(np.sum(np.minimum(a, b)))
        union = float(np.sum(np.maximum(a, b))) + 1e-5
        return inter / union

    @classmethod
    def _template_score(cls, feats: Dict[str, object], tpl: Dict[str, object]) -> float:
        s_gray = cls._similarity(feats["neutral_gray"], tpl["neutral_gray"])
        s_mask = cls._similarity(feats["fg_mask"], tpl["fg_mask"])
        s_iou = cls._mask_iou(feats["fg_mask"], tpl["fg_mask"])
        s_edge = cls._similarity(feats["edges"], tpl["edges"])
        return 0.45 * s_gray + 0.25 * s_mask + 0.15 * s_iou + 0.15 * s_edge

    def recognize(
        self,
        board_image,
        orientation: str = "white",
        side_to_move: str = "w",
        castling: str = "KQkq",
    ) -> RecognitionResult:
        import chess
        import numpy as np

        norm_board = self._normalize_board(board_image)
        est_light, est_dark = self._estimate_backgrounds(norm_board)

        square_feats: Dict[Tuple[int, int], Dict[str, object]] = {}
        occupied_coords: List[Tuple[int, int]] = []
        color_scores: Dict[Tuple[int, int], float] = {}
        highlight_shifts: Dict[Tuple[int, int], float] = {}

        for r in range(8):
            for c in range(8):
                # Always compare against the board's median light/dark background to detect highlighted squares
                bg = est_light if (r + c) % 2 == 0 else est_dark
                patch = self._extract_square_patch(norm_board, r, c)
                feats = self._extract_features(patch, bg)
                square_feats[(r, c)] = feats
                highlight_shifts[(r, c)] = feats["highlight_shift"]

                if feats["is_occupied"]:
                    occupied_coords.append((r, c))
                    color_scores[(r, c)] = feats["body_luma"]

        if color_scores:
            vals = sorted(color_scores.values())
            if (vals[-1] - vals[0]) > 25.0 and len(vals) >= 2:
                # Find the largest gap between consecutive sorted foreground luminance values (Black vs White cluster)
                best_gap = -1.0
                color_threshold = 0.5 * (vals[0] + vals[-1])
                for i in range(len(vals) - 1):
                    gap = vals[i + 1] - vals[i]
                    if gap > best_gap and 40.0 <= 0.5 * (vals[i] + vals[i + 1]) <= 195.0:
                        best_gap = gap
                        color_threshold = 0.5 * (vals[i] + vals[i + 1])
            else:
                color_threshold = 128.0
        else:
            color_threshold = 128.0

        top_lumas = [color_scores[(r, c)] for (r, c) in occupied_coords if r <= 2]
        bottom_lumas = [color_scores[(r, c)] for (r, c) in occupied_coords if r >= 5]
        if top_lumas and bottom_lumas:
            detected_orientation = "white" if float(np.mean(bottom_lumas)) >= float(np.mean(top_lumas)) else "black"
        else:
            detected_orientation = orientation

        eff_orientation = detected_orientation if orientation == "auto" else orientation

        banks = self.template_banks if self.template_banks else ([self.templates] if self.templates else [])
        active_bank: Dict[str, Dict[str, object]] = {}
        if self.calibrated and banks and occupied_coords:
            best_bank_score = -1e9
            for candidate_bank in banks:
                bank_total = 0.0
                for (r, c) in occupied_coords:
                    feats = square_feats[(r, c)]
                    is_white = color_scores[(r, c)] >= color_threshold
                    pool = {sym: tpl for sym, tpl in candidate_bank.items() if sym.isupper() == is_white}
                    if not pool:
                        pool = candidate_bank
                    sq_best = max(
                        0.45 * self._similarity(feats["neutral_gray"], tpl["neutral_gray"])
                        + 0.25 * self._similarity(feats["fg_mask"], tpl["fg_mask"])
                        + 0.15 * self._mask_iou(feats["fg_mask"], tpl["fg_mask"])
                        + 0.15 * self._similarity(feats["edges"], tpl["edges"])
                        for tpl in pool.values()
                    )
                    bank_total += sq_best
                if bank_total > best_bank_score:
                    best_bank_score = bank_total
                    active_bank = candidate_bank

        grid: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]
        sq_candidate_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
        confidences: List[float] = []
        uncertain_squares: List[str] = []

        for r in range(8):
            for c in range(8):
                feats = square_feats[(r, c)]
                file_idx = c if eff_orientation == "white" else 7 - c
                rank_idx = 7 - r if eff_orientation == "white" else r
                sq_name = chess.square_name(chess.square(file_idx, rank_idx))

                if (r, c) not in occupied_coords:
                    confidences.append(0.98)
                    continue

                is_white = color_scores[(r, c)] >= color_threshold
                is_back_rank = r in (0, 7)

                cand_map: Dict[str, float] = {}
                if self.calibrated and active_bank:
                    same_color_pool = {sym: tpl for sym, tpl in active_bank.items() if sym.isupper() == is_white}
                    pool = same_color_pool if same_color_pool else active_bank
                    for sym, tpl in pool.items():
                        if is_back_rank and sym.upper() == "P":
                            continue
                        cand_map[sym] = self._template_score(feats, tpl)
                else:
                    for p_type, syn_mask in self._synthetic_masks.items():
                        if is_back_rank and p_type == "P":
                            continue
                        score = 0.65 * self._similarity(feats["fg_mask"], syn_mask) + 0.35 * self._mask_iou(
                            feats["fg_mask"], syn_mask
                        )
                        cand_map[p_type if is_white else p_type.lower()] = score

                # Template variants learned from manual corrections (other piece sets / themes)
                for sym, samples in self.correction_samples.items():
                    if is_back_rank and sym.upper() == "P":
                        continue
                    for sample in samples:
                        score = self._template_score(feats, sample)
                        if sym.isupper() != is_white and score < 0.85:
                            continue
                        if score > cand_map.get(sym, -1.0):
                            cand_map[sym] = score

                if not cand_map:
                    cand_map = {("Q" if is_white else "q"): 0.0}
                ranked = sorted(cand_map.items(), key=lambda kv: kv[1], reverse=True)
                best_sym, best_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else -1.0
                sq_candidate_scores[(r, c)] = cand_map

                if self.calibrated and active_bank:
                    margin = max(0.0, best_score - second_score)
                    conf = float(np.clip(0.65 * max(0.0, best_score) + 0.35 * min(1.0, margin * 3.0), 0.0, 1.0))
                else:
                    conf = float(np.clip(best_score * 0.80, 0.25, 0.80))
                grid[r][c] = best_sym
                confidences.append(conf)
                if conf < 0.55:
                    uncertain_squares.append(sq_name)

        # Enforce exactly 1 White King ('K') and 1 Black King ('k') so generated FENs are always valid for UCI engines
        for king_sym, is_white_king in (("K", True), ("k", False)):
            color_coords = [
                (r, c)
                for (r, c) in occupied_coords
                if (color_scores[(r, c)] >= color_threshold) == is_white_king
            ]
            if not color_coords:
                continue
            king_coords = [(r, c) for (r, c) in color_coords if grid[r][c] == king_sym]
            if len(king_coords) > 1:
                # Keep King on the square with the strongest King score; demote others to their best non-King piece
                king_coords.sort(
                    key=lambda rc: sq_candidate_scores.get(rc, {}).get(king_sym, -1.0),
                    reverse=True,
                )
                for rc in king_coords[1:]:
                    non_king = [
                        (sc, s)
                        for s, sc in sq_candidate_scores.get(rc, {}).items()
                        if s != king_sym and (rc[0] not in (0, 7) or s.upper() != "P")
                    ]
                    non_king.sort(reverse=True)
                    grid[rc[0]][rc[1]] = non_king[0][1] if non_king else ("Q" if is_white_king else "q")
            elif len(king_coords) == 0:
                best_rc = max(
                    color_coords,
                    key=lambda rc: sq_candidate_scores.get(rc, {}).get(king_sym, -1.0),
                )
                grid[best_rc[0]][best_rc[1]] = king_sym

            # Enforce at most 8 pawns and pawns + promoted <= 8 per color using candidate scores
            pawn_sym = "P" if is_white_king else "p"
            std_limits = (
                {"Q": 1, "R": 2, "B": 2, "N": 2}
                if is_white_king
                else {"q": 1, "r": 2, "b": 2, "n": 2}
            )
            pawn_coords = [(r, c) for (r, c) in color_coords if grid[r][c] == pawn_sym]
            if len(pawn_coords) > 8:
                pawn_coords.sort(key=lambda rc: sq_candidate_scores.get(rc, {}).get(pawn_sym, -1.0))
                for rc in pawn_coords[: len(pawn_coords) - 8]:
                    alts = [
                        (sc, s)
                        for s, sc in sq_candidate_scores.get(rc, {}).items()
                        if s not in (king_sym, pawn_sym)
                    ]
                    alts.sort(reverse=True)
                    grid[rc[0]][rc[1]] = alts[0][1] if alts else ("N" if is_white_king else "n")

            for _ in range(8):
                counts: Dict[str, int] = {}
                for rc in color_coords:
                    s_cur = grid[rc[0]][rc[1]]
                    if s_cur:
                        counts[s_cur] = counts.get(s_cur, 0) + 1
                n_pawns = counts.get(pawn_sym, 0)
                promoted = sum(max(0, counts.get(s, 0) - lim) for s, lim in std_limits.items())
                if n_pawns + promoted <= 8:
                    break
                excess_rcs = [
                    rc
                    for rc in color_coords
                    if grid[rc[0]][rc[1]] in std_limits
                    and counts.get(grid[rc[0]][rc[1]], 0) > std_limits[grid[rc[0]][rc[1]]]  # type: ignore[index]
                ]
                if not excess_rcs:
                    break
                best_swap: Optional[Tuple[float, Tuple[int, int], str]] = None
                for rc in excess_rcs:
                    cur_s = grid[rc[0]][rc[1]]
                    assert cur_s is not None
                    cur_sc = sq_candidate_scores.get(rc, {}).get(cur_s, 0.0)
                    valid_targets = [
                        s for s, lim in std_limits.items() if counts.get(s, 0) < lim
                    ]
                    if n_pawns < 8 and rc[0] not in (0, 7):
                        valid_targets.append(pawn_sym)
                    if not valid_targets:
                        continue
                    for tgt in valid_targets:
                        tgt_sc = sq_candidate_scores.get(rc, {}).get(tgt, -0.5)
                        cost = cur_sc - tgt_sc
                        if best_swap is None or cost < best_swap[0]:
                            best_swap = (cost, rc, tgt)
                if best_swap is None:
                    break
                _, swap_rc, swap_tgt = best_swap
                grid[swap_rc[0]][swap_rc[1]] = swap_tgt

        # Detect visual last-move highlight directly from the board squares (1 empty highlighted + 1 occupied highlighted)
        all_shifts = list(highlight_shifts.values())
        median_shift = float(np.median(all_shifts)) if all_shifts else 0.0
        hl_threshold = max(16.0, 2.5 * median_shift)
        hl_empty: List[Tuple[float, int, int]] = []
        hl_occupied: List[Tuple[float, int, int, str]] = []
        for r in range(8):
            for c in range(8):
                sh = highlight_shifts[(r, c)]
                if sh >= hl_threshold:
                    sym = grid[r][c]
                    if sym is None:
                        hl_empty.append((sh, r, c))
                    else:
                        hl_occupied.append((sh, r, c, sym))

        visual_last_move: Optional[str] = None
        visual_turn: Optional[str] = None
        if hl_empty and hl_occupied:
            hl_empty.sort(reverse=True)
            hl_occupied.sort(reverse=True)
            _, r_from, c_from = hl_empty[0]
            _, r_to, c_to, moved_sym = hl_occupied[0]
            f_from = c_from if eff_orientation == "white" else 7 - c_from
            rk_from = 7 - r_from if eff_orientation == "white" else r_from
            f_to = c_to if eff_orientation == "white" else 7 - c_to
            rk_to = 7 - r_to if eff_orientation == "white" else r_to
            sq_from_name = chess.square_name(chess.square(f_from, rk_from))
            sq_to_name = chess.square_name(chess.square(f_to, rk_to))
            visual_last_move = sq_from_name + sq_to_name
            # If the piece on the highlighted destination square is White, it is now Black's turn ('b'), and vice versa!
            visual_turn = "b" if moved_sym.isupper() else "w"

        board = chess.Board.empty()
        for r in range(8):
            for c in range(8):
                sym = grid[r][c]
                if not sym:
                    continue
                file_idx = c if eff_orientation == "white" else 7 - c
                rank_idx = 7 - r if eff_orientation == "white" else r
                sq = chess.square(file_idx, rank_idx)
                board.set_piece_at(sq, chess.Piece.from_symbol(sym))

        eff_turn = visual_turn if visual_turn in ("w", "b") else side_to_move
        board.turn = chess.WHITE if eff_turn == "w" else chess.BLACK
        if board.was_into_check():
            board.turn = not board.turn
            visual_turn = "w" if board.turn == chess.WHITE else "b"

        # Only set castling rights that are actually valid for the current king/rook positions
        valid_castling = ""
        if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if "K" in castling and board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
                valid_castling += "K"
            if "Q" in castling and board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
                valid_castling += "Q"
        if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if "k" in castling and board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
                valid_castling += "k"
            if "q" in castling and board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
                valid_castling += "q"
        board.set_castling_fen(valid_castling if valid_castling else "-")

        board_fen = board.board_fen()
        full_fen = board.fen()
        overall_conf = float(np.mean(confidences)) if confidences else 0.0

        return RecognitionResult(
            fen=full_fen,
            board_fen=board_fen,
            confidence=overall_conf,
            uncertain_squares=uncertain_squares,
            calibrated=self.calibrated,
            detected_orientation=detected_orientation,
            inferred_turn=visual_turn,
            last_move_uci=visual_last_move,
        )


class UciEngine:
    def __init__(self, executable: str = "", timeout: float = 15.0):
        self.executable = executable or find_default_engine()
        self.timeout = timeout

    def analyse_lines(
        self,
        fen: str,
        depth: int,
        multipv: int = 3,
        movetime_ms: Optional[int] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        stop_event: Optional[threading.Event] = None,
        on_process: Optional[Callable[[object], None]] = None,
    ) -> List[EngineLine]:
        """
        Runs a UCI engine. Uses `go movetime N` when movetime_ms is given ("analyse for N seconds"),
        otherwise `go depth D`. `on_progress` receives human-readable status (depth, nodes/s);
        setting `stop_event` sends `stop` so the engine returns its current best lines immediately.
        """
        import chess

        if not self.executable:
            raise RuntimeError("Informe o caminho de uma engine UCI, por exemplo Stockfish")
        executable = self.executable if os.path.isfile(self.executable) else shutil.which(self.executable)
        if not executable or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Engine nao executavel: {self.executable}")

        # Sanitize FEN so Stockfish never aborts on transient/opposite-check states
        board = chess.Board(fen)
        if board.was_into_check():
            board.turn = not board.turn
        board.castling_rights = board.clean_castling_rights()
        if board.ep_square is not None and not board.has_legal_en_passant():
            board.ep_square = None
        fen = board.fen()
        if not board.is_valid():
            raise ValueError("Posicao FEN invalida para analise UCI")

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        go_cmd = f"go movetime {int(movetime_ms)}" if movetime_ms else f"go depth {depth}"
        commands = [
            "uci",
            f"setoption name MultiPV value {max(1, multipv)}",
            "isready",
            f"position fen {fen}",
            go_cmd,
        ]
        effective_timeout = self.timeout
        if movetime_ms:
            effective_timeout = max(self.timeout, movetime_ms / 1000.0 + 5.0)

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        raw_lines: Dict[int, Tuple[Optional[int], Optional[int], List[str], Optional[int]]] = {}
        best_move: Optional[str] = None
        current_depth = 0
        last_progress = 0.0
        stop_sent = False

        # Avoid `with subprocess.Popen(...)` because Python's Popen.__exit__ calls stdin.close()
        # unguarded, which raises OSError: [Errno 22] Invalid argument on Windows if the child exited early.
        process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        if on_process is not None:
            try:
                on_process(process)
            except Exception:
                pass
        try:
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                for command in commands:
                    process.stdin.write(command + "\n")
                process.stdin.flush()
            except OSError as exc:
                try:
                    process.kill()
                except Exception:
                    pass
                if stop_event is not None and stop_event.is_set():
                    return []
                raise RuntimeError(f"Falha ao enviar comandos para a engine UCI: {exc}") from exc

            line_queue: queue.Queue[Optional[str]] = queue.Queue()

            def _reader(stdout_pipe) -> None:
                try:
                    for raw_line in stdout_pipe:
                        line_queue.put(raw_line)
                except Exception:
                    pass
                finally:
                    line_queue.put(None)

            reader_thread = threading.Thread(target=_reader, args=(process.stdout,), daemon=True)
            reader_thread.start()

            deadline = time.monotonic() + effective_timeout
            timed_out = False
            while True:
                if stop_event is not None and stop_event.is_set() and not stop_sent:
                    stop_sent = True
                    try:
                        process.stdin.write("stop\n")
                        process.stdin.flush()
                    except Exception:
                        pass
                    deadline = min(deadline, time.monotonic() + 0.35)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    item = line_queue.get(timeout=min(0.15, remaining))
                except queue.Empty:
                    if process.poll() is not None and line_queue.empty():
                        break
                    continue

                if item is None:
                    break
                line = item.strip()
                if line.startswith("info"):
                    tokens = line.split()
                    line_depth: Optional[int] = None
                    if "depth" in tokens:
                        try:
                            line_depth = int(tokens[tokens.index("depth") + 1])
                            current_depth = max(current_depth, line_depth)
                        except (ValueError, IndexError):
                            pass
                    if on_progress is not None and time.monotonic() - last_progress > 0.3 and current_depth:
                        last_progress = time.monotonic()
                        nps_txt = ""
                        if "nps" in tokens:
                            try:
                                nps = int(tokens[tokens.index("nps") + 1])
                                nps_txt = f" | {nps / 1_000_000:.1f}M nos/s" if nps >= 1_000_000 else f" | {nps // 1000}k nos/s"
                            except (ValueError, IndexError):
                                pass
                        on_progress(f"{Path(executable).stem}: profundidade {current_depth}{nps_txt}")
                    if " pv " not in f" {line} ":
                        continue
                    mpv = 1
                    if "multipv" in tokens:
                        try:
                            mpv = int(tokens[tokens.index("multipv") + 1])
                        except (ValueError, IndexError):
                            mpv = 1
                    cp_val: Optional[int] = None
                    mate_val: Optional[int] = None
                    if "score" in tokens:
                        s_idx = tokens.index("score")
                        if s_idx + 2 < len(tokens):
                            stype = tokens[s_idx + 1]
                            try:
                                sval = int(tokens[s_idx + 2])
                                if stype == "cp":
                                    cp_val = sval
                                elif stype == "mate":
                                    mate_val = sval
                            except ValueError:
                                pass
                    # Ignore lowerbound/upperbound partial scores (aspiration window fails)
                    if "lowerbound" in tokens or "upperbound" in tokens:
                        continue
                    pv_idx = tokens.index("pv")
                    pv_moves = tokens[pv_idx + 1 :]
                    if pv_moves:
                        raw_lines[mpv] = (cp_val, mate_val, pv_moves, line_depth)
                elif line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != "(none)":
                        best_move = parts[1]
                    break

            if stop_sent or (stop_event is not None and stop_event.is_set()):
                if not best_move:
                    if 1 in raw_lines and raw_lines[1][2]:
                        best_move = raw_lines[1][2][0]
                    elif legal_moves:
                        best_move = legal_moves[0].uci()
            else:
                if timed_out:
                    raise TimeoutError(
                        f"A engine UCI excedeu o tempo limite de {effective_timeout:.1f}s sem retornar bestmove"
                    )
                if not best_move:
                    if 1 in raw_lines and raw_lines[1][2]:
                        best_move = raw_lines[1][2][0]
                    else:
                        raise RuntimeError("A engine nao retornou bestmove")
        finally:
            try:
                if process.poll() is None:
                    try:
                        if process.stdin:
                            process.stdin.write("quit\n")
                            process.stdin.flush()
                    except Exception:
                        pass
                    try:
                        process.wait(timeout=0.25)
                    except Exception:
                        try:
                            process.kill()
                            process.wait(timeout=0.5)
                        except Exception:
                            pass
            except Exception:
                pass
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:
                        pass

        if not raw_lines:
            raw_lines[1] = (None, None, [best_move], current_depth or None)

        results: List[EngineLine] = []
        for mpv in sorted(raw_lines.keys())[: max(1, multipv)]:
            cp_val, mate_val, pv_moves, line_depth = raw_lines[mpv]
            try:
                results.append(
                    build_engine_line(
                        fen=fen,
                        multipv=mpv,
                        pv_uci=pv_moves,
                        score_turn_cp=cp_val,
                        mate_turn=mate_val,
                        depth=line_depth,
                    )
                )
            except Exception:
                continue

        if not results:
            results.append(build_engine_line(fen=fen, multipv=1, pv_uci=[best_move], depth=current_depth or None))
        return results

    def analyse_with_move(self, fen: str, depth: int) -> Tuple[str, str]:
        lines = self.analyse_lines(fen, depth=depth, multipv=1)
        top = lines[0]
        summary = f"Melhor lance: {top.figurine_move} ({top.uci_move}) | avaliacao: {top.eval_text}"
        return top.uci_move, summary

    def analyse(self, fen: str, depth: int) -> str:
        _, summary = self.analyse_with_move(fen, depth)
        return summary


class RegionSelectorOverlay(tk.Toplevel):
    """Fullscreen interactive snipping overlay for selecting the chessboard region with the mouse."""

    def __init__(self, parent: tk.Tk, screen_bgr, on_selected):
        super().__init__(parent)
        self.on_selected = on_selected
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None
        self.label_id = None

        self.overrideredirect(True)
        self.attributes("-topmost", True)
        height, width = screen_bgr.shape[:2]
        self.geometry(f"{width}x{height}+0+0")

        import cv2
        from PIL import Image, ImageTk

        dimmed = (screen_bgr.astype("float32") * 0.65).astype("uint8")
        rgb = cv2.cvtColor(dimmed, cv2.COLOR_BGR2RGB)
        self._bg_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._full_rgb = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2RGB)

        self.canvas = tk.Canvas(self, width=width, height=height, cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self._bg_photo, anchor="nw")
        self.canvas.create_text(
            width // 2,
            34,
            text="Clique e arraste ao redor do tabuleiro de xadrez (ESC para cancelar)",
            fill="#ffffff",
            font=("Segoe UI", 14, "bold"),
        )

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.focus_force()

    def _on_press(self, event) -> None:
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        if self.label_id:
            self.canvas.delete(self.label_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="#38d45a", width=3
        )
        self.label_id = self.canvas.create_text(
            self.start_x,
            max(15, self.start_y - 12),
            text="",
            fill="#38d45a",
            font=("Consolas", 11, "bold"),
            anchor="sw",
        )

    def _on_drag(self, event) -> None:
        if not self.rect_id:
            return
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        self.canvas.coords(self.rect_id, x1, y1, x2, y2)
        if self.label_id:
            self.canvas.coords(self.label_id, x1, max(18, y1 - 6))
            self.canvas.itemconfigure(self.label_id, text=f"{x2 - x1} x {y2 - y1} px")

    def _on_release(self, event) -> None:
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        w, h = x2 - x1, y2 - y1
        self.destroy()
        if w >= 48 and h >= 48:
            self.on_selected(BoardRegion(x1, y1, w, h))
        else:
            self.on_selected(None)

    def _cancel(self) -> None:
        self.destroy()
        self.on_selected(None)


def render_board_bgr(
    fen: str,
    orientation: str = "white",
    size: int = 420,
    uncertain_squares: Sequence[str] = (),
    changed_squares: Sequence[str] = (),
) -> object:
    """Renders a clean 2D BGR chessboard image from a FEN with coordinates and square highlights."""
    import chess
    import cv2
    import numpy as np

    try:
        board = chess.BaseBoard(fen.split()[0])
    except Exception:
        board = chess.BaseBoard()

    base_px = 512
    sq_px = base_px // 8
    img = np.zeros((base_px, base_px, 3), dtype=np.uint8)
    light_bgr = (210, 232, 238)
    dark_bgr = (86, 148, 118)
    masks = PieceRecognizer._build_synthetic_masks()
    uncertain_set = set(uncertain_squares)
    changed_set = set(changed_squares)

    for r in range(8):
        for c in range(8):
            y1, y2 = r * sq_px, (r + 1) * sq_px
            x1, x2 = c * sq_px, (c + 1) * sq_px
            file_idx = c if orientation == "white" else 7 - c
            rank_idx = 7 - r if orientation == "white" else r
            sq = chess.square(file_idx, rank_idx)
            sq_name = chess.square_name(sq)

            is_light = (r + c) % 2 == 0
            bg = light_bgr if is_light else dark_bgr
            img[y1:y2, x1:x2] = bg

            if sq_name in uncertain_set:
                cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (30, 60, 235), 3)
            elif sq_name in changed_set:
                cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (60, 185, 45), 3)

            piece = board.piece_at(sq)
            if piece is not None:
                p_type = piece.symbol().upper()
                mask_52 = masks[p_type]
                m = 6
                sub = img[y1 + m : y2 - m, x1 + m : x2 - m]
                fill_color = (
                    np.array([248, 248, 248], dtype=np.uint8)
                    if piece.color == chess.WHITE
                    else np.array([34, 34, 34], dtype=np.uint8)
                )
                outline_color = (
                    np.array([28, 28, 28], dtype=np.uint8)
                    if piece.color == chess.WHITE
                    else np.array([225, 225, 225], dtype=np.uint8)
                )
                fg = (mask_52 > 0.5).astype(np.uint8)
                eroded = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
                outline = fg - eroded
                sub[eroded > 0] = fill_color
                sub[outline > 0] = outline_color

            # Subtle rank/file coordinates along board edges
            coord_color = dark_bgr if is_light else light_bgr
            if c == 0:
                rank_lbl = str(rank_idx + 1)
                cv2.putText(
                    img, rank_lbl, (x1 + 4, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, coord_color, 1, cv2.LINE_AA
                )
            if r == 7:
                file_lbl = chr(ord("a") + file_idx)
                cv2.putText(
                    img, file_lbl, (x2 - 13, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, coord_color, 1, cv2.LINE_AA
                )

    if size != base_px:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


class ToolTip:
    """Lightweight hover tooltip for Tkinter/ttk widgets."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 350):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: Optional[str] = None
        self._tip_win: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None) -> None:
        self._cancel()
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)
        except Exception:
            pass

    def _on_leave(self, _event=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip_win is not None or not self.text:
            return
        try:
            if not self.widget.winfo_exists():
                return
            x = self.widget.winfo_rootx() + 14
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            win = tk.Toplevel(self.widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{x}+{y}")
            try:
                win.attributes("-topmost", True)
            except Exception:
                pass
            lbl = tk.Label(
                win,
                text=self.text,
                justify="left",
                bg="#1f2937",
                fg="#f9fafb",
                relief="solid",
                bd=1,
                padx=8,
                pady=5,
                font=("Segoe UI", 9),
                wraplength=320,
            )
            lbl.pack()
            self._tip_win = win
        except Exception:
            self._tip_win = None

    def _hide(self) -> None:
        if self._tip_win is not None:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None


def compute_line_confidence(
    line_idx: int,
    lines: Sequence[EngineLine],
    rec_confidence: float = 1.0,
) -> int:
    """
    Computes an intuitive confidence percentage (1..99%) for an engine Multi-PV card,
    combining board recognition confidence with relative evaluation strength.
    """
    import math

    if not lines or line_idx >= len(lines):
        return 0
    base = max(0.55, min(0.99, float(rec_confidence) if rec_confidence > 0 else 0.95))
    top = lines[0]
    curr = lines[line_idx]

    if line_idx == 0:
        if top.mate_white is not None:
            rel = 0.99
        elif len(lines) > 1 and top.score_white_cp is not None and lines[1].score_white_cp is not None:
            gap = abs(top.score_white_cp - lines[1].score_white_cp)
            rel = min(0.99, 0.88 + min(0.10, gap / 250.0))
        else:
            rel = 0.94
    else:
        if top.mate_white is not None and curr.mate_white is None:
            rel = 0.45
        elif top.score_white_cp is not None and curr.score_white_cp is not None:
            gap = abs(top.score_white_cp - curr.score_white_cp)
            rel = max(0.35, 0.92 * math.exp(-gap / 220.0))
        else:
            rel = max(0.50, 0.85 - 0.12 * line_idx)

    pct = int(round(base * rel * 100))
    return max(15, min(99, pct))


class BoardEditorDialog(tk.Toplevel):
    """
    Interactive chessboard editor dialog to inspect and manually fix recognized pieces
    before running engine analysis.
    """

    BOARD_PX = 380

    def __init__(
        self,
        parent: tk.Tk,
        fen: str,
        orientation: str = "white",
        uncertain_squares: Sequence[str] = (),
        learn_default: bool = True,
        on_apply: Optional[Callable[[str, List[str], bool, bool], None]] = None,
    ):
        super().__init__(parent)
        self.title("Editar Tabuleiro")
        self.geometry("720x560")
        self.transient(parent)
        self.grab_set()

        self.model = BoardEditModel(fen)
        self.orientation = orientation
        self.uncertain_squares = list(uncertain_squares)
        self.on_apply = on_apply
        self.selected_tool: Optional[str] = "move"
        self._drag_from_sq: Optional[str] = None
        self._board_photo = None

        main = ttk.Frame(self, padding=14)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", padx=(0, 14))

        self.canvas = tk.Canvas(
            left, width=self.BOARD_PX, height=self.BOARD_PX, bg="#262421", highlightthickness=1
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)

        self.validation_label = tk.Label(
            left, text="", font=("Segoe UI", 9, "bold"), anchor="w", padx=8, pady=5, wraplength=370
        )
        self.validation_label.pack(fill="x", pady=(8, 0))

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Ferramenta de Edicao:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.tool_var = tk.StringVar(value="move")

        tools_frame = ttk.Frame(right)
        tools_frame.pack(fill="x", pady=4)
        ttk.Radiobutton(
            tools_frame, text="Mover / Arrastar pecas", value="move", variable=self.tool_var
        ).pack(anchor="w", pady=1)
        ttk.Radiobutton(
            tools_frame, text="Apagar casa (ou clique direito)", value="erase", variable=self.tool_var
        ).pack(anchor="w", pady=1)

        white_row = ttk.LabelFrame(right, text="Pecas Brancas", padding=6)
        white_row.pack(fill="x", pady=4)
        for sym, label in (("K", "Rei"), ("Q", "Dama"), ("R", "Torre"), ("B", "Bispo"), ("N", "Cav."), ("P", "Peao")):
            ttk.Radiobutton(
                white_row, text=label, value=sym, variable=self.tool_var, width=6
            ).pack(side="left", padx=2)

        black_row = ttk.LabelFrame(right, text="Pecas Pretas", padding=6)
        black_row.pack(fill="x", pady=4)
        for sym, label in (("k", "Rei"), ("q", "Dama"), ("r", "Torre"), ("b", "Bispo"), ("n", "Cav."), ("p", "Peao")):
            ttk.Radiobutton(
                black_row, text=label, value=sym, variable=self.tool_var, width=6
            ).pack(side="left", padx=2)

        state_frame = ttk.LabelFrame(right, text="Turno e Direitos de Roque", padding=6)
        state_frame.pack(fill="x", pady=4)

        turn_row = ttk.Frame(state_frame)
        turn_row.pack(fill="x")
        ttk.Label(turn_row, text="Vez de jogar:").pack(side="left")
        self.turn_var = tk.StringVar(value=self.model.turn)
        ttk.Radiobutton(
            turn_row, text="Brancas", value="w", variable=self.turn_var, command=self._on_meta_change
        ).pack(side="left", padx=8)
        ttk.Radiobutton(
            turn_row, text="Pretas", value="b", variable=self.turn_var, command=self._on_meta_change
        ).pack(side="left")

        castle_row = ttk.Frame(state_frame)
        castle_row.pack(fill="x", pady=(6, 0))
        ttk.Label(castle_row, text="Roque:").pack(side="left")
        self.castle_vars: Dict[str, tk.BooleanVar] = {}
        current_c = self.model.castling
        for flag in ("K", "Q", "k", "q"):
            v = tk.BooleanVar(value=(flag in current_c))
            self.castle_vars[flag] = v
            ttk.Checkbutton(castle_row, text=flag, variable=v, command=self._on_meta_change).pack(
                side="left", padx=3
            )
        ttk.Button(castle_row, text="Auto-roque", command=self._auto_castling).pack(side="left", padx=6)

        util_row = ttk.Frame(right)
        util_row.pack(fill="x", pady=6)
        ttk.Button(util_row, text="Posicao inicial", command=self._reset_start).pack(side="left", padx=(0, 4))
        ttk.Button(util_row, text="Limpar tudo", command=self._clear_board).pack(side="left", padx=4)
        ttk.Button(util_row, text="Inverter lado", command=self._flip_view).pack(side="left", padx=4)

        self.learn_var = tk.BooleanVar(value=learn_default)
        ttk.Checkbutton(
            right,
            text="Aprender pecas corrigidas para futuras capturas",
            variable=self.learn_var,
        ).pack(anchor="w", pady=4)

        ttk.Label(right, text="FEN resultante:").pack(anchor="w")
        self.fen_var = tk.StringVar(value=self.model.to_fen())
        fen_out = tk.Entry(
            right,
            textvariable=self.fen_var,
            state="readonly",
            font=("Consolas", 9),
            readonlybackground="#ffffff",
            fg="#111827",
            relief="solid",
            bd=1,
        )
        fen_out.pack(fill="x", pady=2, ipady=3)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_row, text="Aplicar e Analisar", command=lambda: self._finish(True)).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btn_row, text="Apenas Aplicar", command=lambda: self._finish(False)).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btn_row, text="Cancelar", command=self.destroy).pack(side="right")

        self._refresh()

    def _xy_to_square(self, x: int, y: int) -> Optional[str]:
        import chess

        if not (0 <= x < self.BOARD_PX and 0 <= y < self.BOARD_PX):
            return None
        sq_px = self.BOARD_PX / 8.0
        col = int(x // sq_px)
        row = int(y // sq_px)
        file_idx = col if self.orientation == "white" else 7 - col
        rank_idx = 7 - row if self.orientation == "white" else row
        return chess.square_name(chess.square(file_idx, rank_idx))

    def _on_press(self, event) -> None:
        sq = self._xy_to_square(event.x, event.y)
        if not sq:
            return
        tool = self.tool_var.get()
        if tool == "move":
            self._drag_from_sq = sq
        elif tool == "erase":
            self.model.set_piece(sq, None)
            self._refresh()
        else:
            self.model.toggle_piece(sq, tool)
            self._refresh()

    def _on_release(self, event) -> None:
        if self.tool_var.get() != "move" or not self._drag_from_sq:
            return
        to_sq = self._xy_to_square(event.x, event.y)
        from_sq = self._drag_from_sq
        self._drag_from_sq = None
        if to_sq and to_sq != from_sq:
            self.model.move_piece(from_sq, to_sq)
            self._refresh()

    def _on_right_click(self, event) -> None:
        sq = self._xy_to_square(event.x, event.y)
        if sq:
            self.model.set_piece(sq, None)
            self._refresh()

    def _on_meta_change(self) -> None:
        self.model.set_turn(self.turn_var.get())
        c_str = "".join(flag for flag in ("K", "Q", "k", "q") if self.castle_vars[flag].get())
        self.model.set_castling(c_str or "-")
        self._refresh()

    def _auto_castling(self) -> None:
        self.model.auto_castling()
        c_str = self.model.castling
        for flag in ("K", "Q", "k", "q"):
            self.castle_vars[flag].set(flag in c_str)
        self._refresh()

    def _reset_start(self) -> None:
        self.model.reset_start()
        self.turn_var.set("w")
        for flag in ("K", "Q", "k", "q"):
            self.castle_vars[flag].set(True)
        self._refresh()

    def _clear_board(self) -> None:
        self.model.clear()
        self._refresh()

    def _flip_view(self) -> None:
        self.orientation = "black" if self.orientation == "white" else "white"
        self._refresh()

    def _refresh(self) -> None:
        import cv2
        from PIL import Image, ImageTk

        fen = self.model.to_fen()
        self.fen_var.set(fen)
        report = self.model.validate()
        if report.ok and not report.warnings:
            self.validation_label.configure(text="🟢 Posicao valida", bg="#d4edda", fg="#155724")
        elif report.ok:
            self.validation_label.configure(text=f"🟡 {report.summary()}", bg="#fff3cd", fg="#856404")
        else:
            self.validation_label.configure(text=f"🔴 {report.summary()}", bg="#f8d7da", fg="#721c24")

        bgr = render_board_bgr(
            fen,
            orientation=self.orientation,
            size=self.BOARD_PX,
            uncertain_squares=self.uncertain_squares,
            changed_squares=self.model.changed_squares(),
        )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._board_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._board_photo, anchor="nw")

    def _finish(self, run_analysis: bool) -> None:
        fen = self.model.to_fen()
        changed = self.model.changed_squares()
        learn = bool(self.learn_var.get())
        self.destroy()
        if self.on_apply:
            self.on_apply(fen, changed, learn, run_analysis)


class ChessCoachApp(tk.Tk):
    BOARD_PX = 420
    EVAL_BAR_W = 26

    def __init__(self, engine_path: str = "", settings_path: Path = DEFAULT_SETTINGS_PATH):
        super().__init__()
        self.title("Chess Coach")
        self.settings_path = Path(settings_path)
        self.settings = load_settings(self.settings_path)
        self.geometry(self.settings.window_geometry or "1140x820")
        self.minsize(980, 720)

        self.capture = ScreenCapture()
        self.detector = BoardDetector()
        self.recognizer = PieceRecognizer()
        self.history = PositionHistory()
        self.training_session = TrainingSession()

        self.last_image = None
        self.last_board_image = None
        self.last_uncertain_squares: List[str] = []
        self.last_recognition_confidence: float = 1.0
        self.region: Optional[BoardRegion] = None
        self.preview_photo = None
        self.piece_icons: Dict[str, object] = {}
        self.last_lines: List[EngineLine] = []
        self.last_best_move: Optional[str] = None
        self.last_detected_move: Optional[str] = None
        self.prev_board_fen: Optional[str] = None
        self.last_analysis_text: str = ""
        self.training_hits = 0
        self.training_misses = 0
        self.monitoring_job: Optional[str] = None
        self._analysis_seq: int = 0
        self._is_analysing: bool = False
        self._stop_event: Optional[threading.Event] = None
        self._active_engine_process: Optional[object] = None
        self._editor_dialog: Optional[BoardEditorDialog] = None
        self._ui_queue: queue.Queue[Tuple[int, str, object]] = queue.Queue()
        self._inline_drag_from: Optional[str] = None
        self._quick_settings_visible: bool = False
        self._tooltips: List[ToolTip] = []

        self._init_piece_icons()
        effective_engine = engine_path or self.settings.engine_path or find_default_engine()
        self._build_ui(effective_engine)
        self._apply_loaded_settings()
        self.history.add(self.fen_var.get().strip(), source="inicial")
        self._update_history_ui()
        self._update_validation_badge()
        self._update_turn_banners()
        self._update_preview(None, draw_arrows=False)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_ui_queue)

    def _add_tip(self, widget: tk.Widget, text: str) -> None:
        self._tooltips.append(ToolTip(widget, text))

    def _init_piece_icons(self) -> None:
        """Generates 28x28 graphical piece icons (PhotoImage) for all 12 chess pieces."""
        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageTk

            masks = PieceRecognizer._build_synthetic_masks()
            for p_type, mask_52 in masks.items():
                fg = (mask_52 > 0.5).astype(np.uint8)
                eroded = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
                outline = fg - eroded

                for is_white in (True, False):
                    sym = p_type if is_white else p_type.lower()
                    bg_rgb = (118, 150, 86) if is_white else (238, 238, 210)
                    canvas = np.full((52, 52, 3), bg_rgb, dtype=np.uint8)
                    fill = (250, 250, 250) if is_white else (36, 36, 36)
                    border = (30, 30, 30) if is_white else (235, 235, 235)
                    canvas[eroded > 0] = fill
                    canvas[outline > 0] = border
                    pil_img = Image.fromarray(canvas).resize((28, 28), Image.Resampling.LANCZOS)
                    self.piece_icons[sym] = ImageTk.PhotoImage(pil_img)
        except Exception:
            self.piece_icons = {}

    def _format_calib_badge(self) -> str:
        if not self.recognizer.calibrated:
            return "Padrao (sem calibracao)"
        themes = self.recognizer.theme_count
        corrs = self.recognizer.correction_count
        extra = f" + {corrs} ajustes" if corrs else ""
        return f"Calibrado ({themes} tema(s){extra})"

    def _set_status_indicator(self, state: str, subtitle: str = "") -> None:
        """
        Updates the compact status indicator pill:
        - 'ready':     🟢 Pronto
        - 'your_turn': 🟢 Sua vez
        - 'analysing': 🟡 Analisando...
        - 'waiting':   🔵 Aguardando jogada
        - 'error':     🔴 Erro
        """
        styles = {
            "ready": ("🟢 Pronto", "#dcfce7", "#166534"),
            "your_turn": ("🟢 Sua vez", "#dcfce7", "#166534"),
            "analysing": ("🟡 Analisando...", "#fef9c3", "#854d0e"),
            "waiting": ("🔵 Aguardando jogada", "#dbeafe", "#1e40af"),
            "error": ("🔴 Erro", "#fee2e2", "#991b1b"),
        }
        badge_text, bg_col, fg_col = styles.get(state, styles["ready"])
        self.turn_state_banner.configure(text=f" {badge_text} ", bg=bg_col, fg=fg_col)
        if subtitle:
            self.turn_banner_var.set(subtitle)

    def toggle_quick_settings(self) -> None:
        """Toggles the collapsible quick settings strip on the main tab."""
        self._quick_settings_visible = not self._quick_settings_visible
        if self._quick_settings_visible:
            self.quick_settings_frame.pack(fill="x", pady=(8, 4), before=self.bottom_strip_frame)
            self.btn_toggle_settings.configure(text="Ocultar configuracoes")
        else:
            self.quick_settings_frame.pack_forget()
            self.btn_toggle_settings.configure(text="Configuracoes")

    def _build_ui(self, engine_path: str) -> None:
        style = ttk.Style(self)
        style.configure("TNotebook.Tab", padding=(14, 6), font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TCheckbutton", font=("Segoe UI", 10))
        style.configure("TRadiobutton", font=("Segoe UI", 10))

        # Shared state variables across tabs
        self.topmost_var = tk.BooleanVar(value=self.settings.topmost)
        self.calib_badge = tk.StringVar(value=self._format_calib_badge())
        self.region_var = tk.StringVar(value=self.settings.region)
        self.orientation_var = tk.StringVar(value=self.settings.orientation)
        self.turn_var = tk.StringVar(value="w")
        self.auto_orient_var = tk.BooleanVar(value=self.settings.auto_orient)
        self.auto_turn_var = tk.BooleanVar(value=self.settings.auto_turn)
        self.review_before_analysis_var = tk.BooleanVar(value=self.settings.review_before_analysis)
        self.view_mode_var = tk.BooleanVar(value=False)
        self.inline_tool_var = tk.StringVar(value="move")
        self.fen_var = tk.StringVar(value=DEFAULT_START_FEN)
        self.engine_var = tk.StringVar(value=engine_path)
        self.analysis_mode_var = tk.StringVar(value=self.settings.analysis_mode)
        self.depth_var = tk.IntVar(value=self.settings.depth)
        self.movetime_var = tk.DoubleVar(value=self.settings.movetime_sec)
        self.auto_monitor_var = tk.BooleanVar(value=self.settings.auto_monitor)
        self.training_mode_var = tk.BooleanVar(value=self.settings.training_mode)
        self.tolerance_cp_var = tk.IntVar(value=self.settings.accept_tolerance_cp)
        self.guess_var = tk.StringVar()
        self.score_label_var = tk.StringVar(value="Acertos: 0 | Erros: 0")
        self.confidence_var = tk.StringVar(value="Confianca do reconhecimento: -")
        self.validation_var = tk.StringVar(value="Posicao valida")
        self.turn_banner_var = tk.StringVar(value="Brancas jogam")
        self.history_pos_var = tk.StringVar(value="Posicao 1/1")
        self.recent_moves_var = tk.StringVar(value="Ultimos lances: (posicao inicial)")
        self.engine_status_var = tk.StringVar(value="Engine pronta")
        self.status = tk.StringVar(
            value="Pressione 'Capturar e analisar (F8)' para analisar o tabuleiro da tela."
        )

        root_frame = ttk.Frame(self, padding=14)
        root_frame.pack(fill="both", expand=True)

        # ── Top Header Bar: | Chess Coach                 [Configuracoes] | ──
        header = ttk.Frame(root_frame)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Chess Coach", font=("Segoe UI", 18, "bold")).pack(side="left")

        self.btn_toggle_settings = ttk.Button(
            header, text="Configuracoes", command=self.toggle_quick_settings
        )
        self.btn_toggle_settings.pack(side="right", padx=(8, 0))
        self._add_tip(
            self.btn_toggle_settings,
            "Mostrar ou ocultar a barra rapida de FEN, orientacao, turno, engine e profundidade.",
        )

        chk_top = ttk.Checkbutton(
            header,
            text="Fixar no topo",
            variable=self.topmost_var,
            command=self._toggle_topmost,
        )
        chk_top.pack(side="right", padx=(10, 4))
        self._add_tip(chk_top, "Mantem a janela do Chess Coach visivel sobre o navegador.")

        ttk.Label(header, textvariable=self.calib_badge, foreground="#4b5563").pack(side="right", padx=8)

        # ── Main Notebook (4 organized tabs) ──
        self.notebook = ttk.Notebook(root_frame)
        self.notebook.pack(fill="both", expand=True)

        tab_play = ttk.Frame(self.notebook, padding=12)
        tab_capture = ttk.Frame(self.notebook, padding=14)
        tab_engine = ttk.Frame(self.notebook, padding=14)
        tab_train = ttk.Frame(self.notebook, padding=14)

        self.notebook.add(tab_play, text="  Jogar / Analisar  ")
        self.notebook.add(tab_capture, text="  Captura e reconhecimento  ")
        self.notebook.add(tab_engine, text="  Engine  ")
        self.notebook.add(tab_train, text="  Treino  ")

        # =====================================================================
        # TAB 1: JOGAR / ANALISAR (Central Board + Analysis Cards)
        # =====================================================================
        upper_split = ttk.Frame(tab_play)
        upper_split.pack(fill="both", expand=True)

        # Left/Center Column: BIG CHESSBOARD + Primary CTA
        board_col = ttk.Frame(upper_split)
        board_col.pack(side="left", anchor="n", padx=(0, 18))

        board_wrap = tk.Frame(board_col, bg="#1f2937", padx=3, pady=3)
        board_wrap.pack()

        self.eval_bar_canvas = tk.Canvas(
            board_wrap,
            width=self.EVAL_BAR_W,
            height=self.BOARD_PX,
            bg="#262421",
            highlightthickness=0,
        )
        self.eval_bar_canvas.pack(side="left", padx=(0, 4))
        self._draw_eval_bar(0, None)

        self.board_canvas = tk.Canvas(
            board_wrap,
            width=self.BOARD_PX,
            height=self.BOARD_PX,
            bg="#262421",
            highlightthickness=0,
            cursor="hand2",
        )
        self.board_canvas.pack(side="left")
        self.board_canvas.bind("<ButtonPress-1>", self._on_inline_board_press)
        self.board_canvas.bind("<ButtonRelease-1>", self._on_inline_board_release)
        self.board_canvas.bind("<ButtonPress-3>", self._on_inline_board_right_click)
        self.preview_label = self.board_canvas
        self._add_tip(
            self.board_canvas,
            "Tabuleiro interativo: arraste uma peca para move-la ou clique com o botao direito para apagar uma casa.",
        )

        # Prominent Primary Action Button: [Capturar e analisar]
        self.btn_primary_capture = tk.Button(
            board_col,
            text="Capturar e analisar (F8)",
            command=self.full_pipeline,
            font=("Segoe UI", 12, "bold"),
            bg="#15803d",
            fg="#ffffff",
            activebackground="#166534",
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=9,
            cursor="hand2",
        )
        self.btn_primary_capture.pack(fill="x", pady=(10, 6))
        self._add_tip(
            self.btn_primary_capture,
            "Captura a tela, reconhece a posicao do tabuleiro automaticamente e inicia a analise do Stockfish (Atalho: F8).",
        )

        board_sub_actions = ttk.Frame(board_col)
        board_sub_actions.pack(fill="x")
        btn_reanalyse = ttk.Button(board_sub_actions, text="Analisar posicao", command=self.analyse)
        btn_reanalyse.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self._add_tip(btn_reanalyse, "Analisa a FEN atual sem capturar novamente a tela.")

        btn_flip_side = ttk.Button(board_sub_actions, text="Inverter lado", command=self.flip_orientation)
        btn_flip_side.pack(side="left", expand=True, fill="x", padx=3)
        self._add_tip(btn_flip_side, "Alterna a orientacao do tabuleiro entre Brancas embaixo e Pretas embaixo.")

        btn_flip_turn = ttk.Button(board_sub_actions, text="Trocar vez", command=self.flip_turn)
        btn_flip_turn.pack(side="left", expand=True, fill="x", padx=3)
        self._add_tip(btn_flip_turn, "Alterna de quem e a vez de jogar (Brancas ou Pretas) e reanalisa.")

        btn_open_ed = ttk.Button(board_sub_actions, text="Editar tabuleiro", command=self.open_board_editor)
        btn_open_ed.pack(side="left", expand=True, fill="x", padx=(3, 0))
        self._add_tip(btn_open_ed, "Abre o editor visual completo para corrigir qualquer peca ou roque.")

        # Right Column: Compact Status Pill + Multi-PV Analysis Cards
        analysis_col = ttk.Frame(upper_split)
        analysis_col.pack(side="left", fill="both", expand=True)

        status_bar = ttk.Frame(analysis_col)
        status_bar.pack(fill="x", pady=(0, 10))

        self.turn_state_banner = tk.Label(
            status_bar,
            text=" 🟢 Pronto ",
            font=("Segoe UI", 11, "bold"),
            bg="#dcfce7",
            fg="#166534",
            padx=10,
            pady=4,
        )
        self.turn_state_banner.pack(side="left")

        ttk.Label(
            status_bar,
            textvariable=self.turn_banner_var,
            font=("Segoe UI", 11, "bold"),
            foreground="#1f2937",
        ).pack(side="left", padx=(10, 6))

        self.validation_badge = tk.Label(
            status_bar,
            textvariable=self.validation_var,
            font=("Segoe UI", 9, "bold"),
            bg="#dcfce7",
            fg="#166534",
            padx=8,
            pady=3,
        )
        self.validation_badge.pack(side="right")

        btn_stop = ttk.Button(status_bar, text="Parar", width=7, command=self.stop_analysis)
        btn_stop.pack(side="right", padx=(0, 6))
        self._add_tip(btn_stop, "Interrompe a busca atual da engine e exibe o melhor lance encontrado ate agora.")

        # Analysis section header
        lines_header = ttk.Frame(analysis_col)
        lines_header.pack(fill="x", pady=(2, 6))
        self.lines_title_var = tk.StringVar(value="Melhores lances")
        ttk.Label(lines_header, textvariable=self.lines_title_var, font=("Segoe UI", 12, "bold")).pack(
            side="left"
        )
        ttk.Label(lines_header, textvariable=self.engine_status_var, foreground="#4b5563").pack(side="right")

        # Container for the 3 Multi-PV Cards
        self.lines_frame = ttk.Frame(analysis_col)
        self.lines_frame.pack(fill="x")
        self.line_rows: List[Dict[str, object]] = []

        for i in range(3):
            card = tk.Frame(
                self.lines_frame,
                bg="#ffffff",
                highlightbackground="#d1d5db",
                highlightcolor="#9ca3af",
                highlightthickness=1,
                padx=12,
                pady=10,
            )
            card.pack(fill="x", pady=5)

            top_row = tk.Frame(card, bg="#ffffff")
            top_row.pack(fill="x")

            rank_lbl = tk.Label(
                top_row,
                text=f"{i + 1}.",
                font=("Segoe UI", 13, "bold"),
                bg="#ffffff",
                fg="#374151",
            )
            rank_lbl.pack(side="left", padx=(0, 4))

            icon_lbl = tk.Label(top_row, bg="#ffffff")
            icon_lbl.pack(side="left", padx=(0, 6))

            move_lbl = tk.Label(
                top_row,
                text="Aguardando analise" if i == 0 else "-",
                font=("Segoe UI Symbol", 13, "bold"),
                bg="#ffffff",
                fg="#111827",
                anchor="w",
            )
            move_lbl.pack(side="left", padx=(0, 10))

            eval_lbl = tk.Label(
                top_row,
                text=" 0.00 ",
                font=("Consolas", 11, "bold"),
                bg="#f3f4f6",
                fg="#111827",
                padx=8,
                pady=2,
                relief="solid",
                bd=1,
            )
            eval_lbl.pack(side="left", padx=(0, 10))

            conf_lbl = tk.Label(
                top_row,
                text="",
                font=("Segoe UI", 9, "bold"),
                bg="#ffffff",
                fg="#047857",
            )
            conf_lbl.pack(side="left")

            exp_lbl = tk.Label(
                top_row,
                text="",
                font=("Segoe UI", 9, "italic"),
                bg="#ffffff",
                fg="#166534",
            )
            exp_lbl.pack(side="right")

            pv_lbl = tk.Label(
                card,
                text="Variacao principal: -" if i == 0 else "",
                font=("Segoe UI Symbol", 10),
                bg="#ffffff",
                fg="#4b5563",
                anchor="w",
                justify="left",
                wraplength=540,
            )
            pv_lbl.pack(fill="x", pady=(6, 0))

            self.line_rows.append({
                "card": card,
                "rank": rank_lbl,
                "eval": eval_lbl,
                "icon": icon_lbl,
                "move": move_lbl,
                "conf": conf_lbl,
                "pv": pv_lbl,
                "exp": exp_lbl,
            })

        # Legend & recognition confidence under cards
        legend_row = ttk.Frame(analysis_col)
        legend_row.pack(fill="x", pady=(8, 0))
        ttk.Label(
            legend_row,
            text="Setas no tabuleiro: Verde = 1º lance | Azul = 2º lance | Amarelo = ultimo lance jogado",
            foreground="#4b5563",
            font=("Segoe UI", 9),
        ).pack(side="left")
        ttk.Label(
            legend_row,
            textvariable=self.confidence_var,
            foreground="#374151",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")

        # ── Collapsible Quick Settings Strip: | FEN | orientacao | turno | engine | profundidade | ──
        self.quick_settings_frame = ttk.LabelFrame(
            tab_play, text="Configuracoes Rapidas (FEN | Orientacao | Turno | Profundidade)", padding=10
        )
        qs_row1 = ttk.Frame(self.quick_settings_frame)
        qs_row1.pack(fill="x", pady=(0, 6))
        ttk.Label(qs_row1, text="FEN:").pack(side="left")
        qs_fen_entry = tk.Entry(
            qs_row1,
            textvariable=self.fen_var,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            relief="solid",
            bd=1,
        )
        qs_fen_entry.pack(side="left", fill="x", expand=True, padx=6, ipady=3)
        qs_fen_entry.bind("<Return>", lambda _e: self.apply_manual_fen())
        ttk.Button(qs_row1, text="Aplicar FEN", command=self.apply_manual_fen).pack(side="left", padx=2)
        ttk.Button(qs_row1, text="Copiar FEN", command=self.copy_fen).pack(side="left", padx=2)
        ttk.Button(qs_row1, text="Copiar analise", command=self.copy_analysis).pack(side="left", padx=2)

        qs_row2 = ttk.Frame(self.quick_settings_frame)
        qs_row2.pack(fill="x")
        ttk.Label(qs_row2, text="Orientacao:").pack(side="left")
        ttk.Combobox(
            qs_row2,
            textvariable=self.orientation_var,
            values=("white", "black"),
            state="readonly",
            width=7,
        ).pack(side="left", padx=4)
        ttk.Button(qs_row2, text="Inverter lado", command=self.flip_orientation).pack(side="left", padx=(0, 12))

        ttk.Label(qs_row2, text="Turno:").pack(side="left")
        ttk.Combobox(
            qs_row2,
            textvariable=self.turn_var,
            values=("w", "b"),
            state="readonly",
            width=4,
        ).pack(side="left", padx=4)
        ttk.Button(qs_row2, text="Trocar vez", command=self.flip_turn).pack(side="left", padx=(0, 12))

        ttk.Label(qs_row2, text="Profundidade:").pack(side="left")
        ttk.Spinbox(qs_row2, from_=1, to=30, textvariable=self.depth_var, width=4).pack(side="left", padx=4)

        ttk.Checkbutton(
            qs_row2,
            text="Monitorar tela automaticamente (1.5s)",
            variable=self.auto_monitor_var,
            command=self.toggle_monitoring,
        ).pack(side="left", padx=12)

        # ── Bottom Strip: | Treino / historico / mensagens | ──
        self.bottom_strip_frame = ttk.Frame(tab_play)
        self.bottom_strip_frame.pack(fill="x", pady=(10, 0))

        hist_bar = ttk.LabelFrame(self.bottom_strip_frame, text="Historico e Treino Rapido", padding=8)
        hist_bar.pack(fill="x")

        h_row = ttk.Frame(hist_bar)
        h_row.pack(fill="x")
        self.btn_first = ttk.Button(h_row, text="Inicio", width=6, command=self.history_first)
        self.btn_first.pack(side="left")
        self.btn_back = ttk.Button(h_row, text="Voltar", width=7, command=self.history_back)
        self.btn_back.pack(side="left", padx=3)
        self.btn_forward = ttk.Button(h_row, text="Avancar", width=8, command=self.history_forward)
        self.btn_forward.pack(side="left", padx=3)
        self.btn_last = ttk.Button(h_row, text="Fim", width=5, command=self.history_last)
        self.btn_last.pack(side="left")

        ttk.Label(h_row, textvariable=self.history_pos_var, font=("Consolas", 10, "bold")).pack(
            side="left", padx=10
        )
        ttk.Label(
            h_row,
            textvariable=self.recent_moves_var,
            font=("Segoe UI Symbol", 10),
            foreground="#374151",
        ).pack(side="left", fill="x", expand=True, padx=6)

        ttk.Button(h_row, text="Salvar PGN", command=self.export_pgn).pack(side="right", padx=2)
        ttk.Button(h_row, text="Copiar analise", command=self.copy_analysis).pack(side="right", padx=2)

        qtrain_row = ttk.Frame(hist_bar)
        qtrain_row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            qtrain_row,
            text="Modo treino (ocultar lances)",
            variable=self.training_mode_var,
            command=self._refresh_analysis_display,
        ).pack(side="left")
        ttk.Label(qtrain_row, text="Seu lance:").pack(side="left", padx=(14, 4))
        q_guess = tk.Entry(
            qtrain_row,
            textvariable=self.guess_var,
            width=12,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#111827",
            relief="solid",
            bd=1,
        )
        q_guess.pack(side="left", padx=4, ipady=2)
        q_guess.bind("<Return>", lambda _e: self.check_guess())
        ttk.Button(qtrain_row, text="Verificar lance", command=self.check_guess).pack(side="left", padx=2)
        ttk.Button(qtrain_row, text="Revelar resposta", command=self.reveal_best_move).pack(side="left", padx=4)
        ttk.Label(qtrain_row, textvariable=self.score_label_var, font=("Segoe UI", 9, "bold")).pack(
            side="right"
        )

        # =====================================================================
        # TAB 2: CAPTURA E RECONHECIMENTO (Advanced Capture & Board Controls)
        # =====================================================================
        cap_group = ttk.LabelFrame(tab_capture, text="Captura de Tela e Calibracao", padding=12)
        cap_group.pack(fill="x", pady=(0, 12))

        cap_btns = ttk.Frame(cap_group)
        cap_btns.pack(fill="x", pady=(0, 10))

        b_sel = ttk.Button(cap_btns, text="Selecionar area com mouse", command=self.select_region_with_mouse)
        b_sel.pack(side="left", padx=(0, 6))
        self._add_tip(b_sel, "Permite desenhar um retangulo com o mouse exatamente sobre o tabuleiro na tela.")

        b_cap = ttk.Button(cap_btns, text="Capturar tela", command=self.capture_screen)
        b_cap.pack(side="left", padx=6)
        self._add_tip(b_cap, "Captura uma imagem da tela inteira sem iniciar a analise.")

        b_det = ttk.Button(cap_btns, text="Detectar tabuleiro automaticamente", command=self.detect_board)
        b_det.pack(side="left", padx=6)
        self._add_tip(b_det, "Localiza automaticamente as bordas do tabuleiro na imagem capturada.")

        b_rec = ttk.Button(cap_btns, text="Reconhecer pecas", command=self.recognize_board)
        b_rec.pack(side="left", padx=6)
        self._add_tip(b_rec, "Reconhece as pecas na regiao do tabuleiro e gera a FEN correspondente.")

        b_cal = ttk.Button(cap_btns, text="Calibrar posicao inicial", command=self.calibrate_from_screen)
        b_cal.pack(side="left", padx=6)
        self._add_tip(
            b_cal,
            "Salva os modelos visuais das 12 pecas a partir de um tabuleiro na posicao inicial para este tema.",
        )

        reg_row = ttk.Frame(cap_group)
        reg_row.pack(fill="x", pady=4)
        ttk.Label(reg_row, text="Regiao do tabuleiro (x, y, largura, altura):").pack(side="left")
        reg_entry = tk.Entry(
            reg_row,
            textvariable=self.region_var,
            width=22,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#111827",
            relief="solid",
            bd=1,
        )
        reg_entry.pack(side="left", padx=8, ipady=3)
        self._add_tip(reg_entry, "Coordenadas em pixels da area do tabuleiro na tela (ex: 2349,234,900,900).")

        rec_opts = ttk.LabelFrame(tab_capture, text="Opcoes de Reconhecimento e Orientacao", padding=12)
        rec_opts.pack(fill="x", pady=(0, 12))

        ro_row1 = ttk.Frame(rec_opts)
        ro_row1.pack(fill="x", pady=4)
        ttk.Label(ro_row1, text="Meu lado no tabuleiro:").pack(side="left")
        ttk.Combobox(
            ro_row1,
            textvariable=self.orientation_var,
            values=("white", "black"),
            state="readonly",
            width=8,
        ).pack(side="left", padx=6)
        ttk.Button(ro_row1, text="Inverter lado", command=self.flip_orientation).pack(side="left", padx=6)

        ttk.Label(ro_row1, text="Vez de jogar:").pack(side="left", padx=(18, 4))
        ttk.Combobox(
            ro_row1,
            textvariable=self.turn_var,
            values=("w", "b"),
            state="readonly",
            width=5,
        ).pack(side="left", padx=4)
        ttk.Button(ro_row1, text="Trocar vez", command=self.flip_turn).pack(side="left", padx=6)

        ro_row2 = ttk.Frame(rec_opts)
        ro_row2.pack(fill="x", pady=(8, 2))
        chk_ao = ttk.Checkbutton(
            ro_row2,
            text="Detectar orientacao automaticamente",
            variable=self.auto_orient_var,
        )
        chk_ao.pack(side="left", padx=(0, 16))
        self._add_tip(chk_ao, "Identifica se voce esta jogando de Brancas ou Pretas pela posicao das pecas.")

        chk_at = ttk.Checkbutton(
            ro_row2,
            text="Detectar vez automaticamente pelo ultimo lance",
            variable=self.auto_turn_var,
        )
        chk_at.pack(side="left", padx=(0, 16))
        self._add_tip(chk_at, "Usa as casas destacadas do ultimo lance e a diferenca de posicao para saber de quem e a vez.")

        chk_rev = ttk.Checkbutton(
            ro_row2,
            text="Revisar tabuleiro antes de analisar",
            variable=self.review_before_analysis_var,
        )
        chk_rev.pack(side="left")
        self._add_tip(chk_rev, "Abre o editor de tabuleiro apos cada captura para confirmar as pecas antes da analise.")

        ro_row3 = ttk.Frame(rec_opts)
        ro_row3.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            ro_row3,
            text="Mostrar diagrama 2D limpo no tabuleiro principal (em vez da captura da tela)",
            variable=self.view_mode_var,
            command=lambda: self._update_preview(self.last_board_image, draw_arrows=True),
        ).pack(side="left")

        edit_group = ttk.LabelFrame(
            tab_capture, text="Edicao Manual e FEN da Posicao", padding=12
        )
        edit_group.pack(fill="x")

        pal_row = ttk.Frame(edit_group)
        pal_row.pack(fill="x", pady=(0, 8))
        ttk.Label(pal_row, text="Ferramenta ao clicar no tabuleiro:").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(pal_row, text="Mover peca", value="move", variable=self.inline_tool_var).pack(
            side="left", padx=4
        )
        ttk.Radiobutton(pal_row, text="Apagar casa", value="erase", variable=self.inline_tool_var).pack(
            side="left", padx=4
        )
        for sym, lbl in (("K", "R"), ("Q", "D"), ("R", "T"), ("B", "B"), ("N", "C"), ("P", "P")):
            ttk.Radiobutton(
                pal_row, text=f"{lbl}B", value=sym, variable=self.inline_tool_var, width=3
            ).pack(side="left", padx=1)
        for sym, lbl in (("k", "R"), ("q", "D"), ("r", "T"), ("b", "B"), ("n", "C"), ("p", "P")):
            ttk.Radiobutton(
                pal_row, text=f"{lbl}P", value=sym, variable=self.inline_tool_var, width=3
            ).pack(side="left", padx=1)
        ttk.Button(pal_row, text="Abrir Editor Completo", command=self.open_board_editor).pack(side="right")

        fen_tab_row = ttk.Frame(edit_group)
        fen_tab_row.pack(fill="x", pady=(4, 0))
        ttk.Label(fen_tab_row, text="FEN:").pack(side="left")
        fen_entry_tab = tk.Entry(
            fen_tab_row,
            textvariable=self.fen_var,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            relief="solid",
            bd=1,
        )
        fen_entry_tab.pack(side="left", fill="x", expand=True, padx=8, ipady=4)
        fen_entry_tab.bind("<Return>", lambda _e: self.apply_manual_fen())
        ttk.Button(fen_tab_row, text="Aplicar FEN", command=self.apply_manual_fen).pack(side="left", padx=3)
        ttk.Button(fen_tab_row, text="Copiar FEN", command=self.copy_fen).pack(side="left", padx=3)

        # =====================================================================
        # TAB 3: ENGINE (UCI Path, Search Depth / Time Limit, Monitoring)
        # =====================================================================
        eng_box = ttk.LabelFrame(tab_engine, text="Configuracao da Engine UCI (Stockfish)", padding=14)
        eng_box.pack(fill="x", pady=(0, 14))

        ttk.Label(eng_box, text="Executavel da Engine UCI (Stockfish detectado automaticamente):").pack(
            anchor="w", pady=(0, 6)
        )
        eng_path_row = ttk.Frame(eng_box)
        eng_path_row.pack(fill="x")
        eng_entry = tk.Entry(
            eng_path_row,
            textvariable=self.engine_var,
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            relief="solid",
            bd=1,
        )
        eng_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=4)
        self._add_tip(eng_entry, "Caminho completo para o executavel do Stockfish ou outra engine compativel com UCI.")
        ttk.Button(eng_path_row, text="Procurar...", command=self.browse_engine).pack(side="left")

        limits_box = ttk.LabelFrame(tab_engine, text="Limite de Busca e Monitoramento", padding=14)
        limits_box.pack(fill="x", pady=(0, 14))

        lim_row1 = ttk.Frame(limits_box)
        lim_row1.pack(fill="x", pady=4)
        rb_depth = ttk.Radiobutton(
            lim_row1,
            text="Analisar por profundidade fixa (plies):",
            value="depth",
            variable=self.analysis_mode_var,
        )
        rb_depth.pack(side="left")
        sp_depth = ttk.Spinbox(lim_row1, from_=1, to=30, textvariable=self.depth_var, width=5)
        sp_depth.pack(side="left", padx=(8, 24))
        self._add_tip(sp_depth, "Profundidade da busca (recomendado: 12 a 18 para analise rapida e precisa).")

        rb_time = ttk.Radiobutton(
            lim_row1,
            text="Analisar por tempo limite (segundos):",
            value="time",
            variable=self.analysis_mode_var,
        )
        rb_time.pack(side="left")
        sp_time = ttk.Spinbox(
            lim_row1, from_=0.5, to=60.0, increment=0.5, textvariable=self.movetime_var, width=6
        )
        sp_time.pack(side="left", padx=8)
        self._add_tip(sp_time, "Tempo maximo em segundos para cada analise (ex: 5.0 segundos).")

        lim_row2 = ttk.Frame(limits_box)
        lim_row2.pack(fill="x", pady=(12, 4))
        chk_mon = ttk.Checkbutton(
            lim_row2,
            text="Monitorar mudancas na tela a cada 1.5s e atualizar automaticamente",
            variable=self.auto_monitor_var,
            command=self.toggle_monitoring,
        )
        chk_mon.pack(side="left")
        self._add_tip(
            chk_mon,
            "Observa continuamente a regiao do tabuleiro e dispara a analise assim que um novo lance e jogado.",
        )

        eng_actions = ttk.Frame(tab_engine)
        eng_actions.pack(fill="x")
        ttk.Button(eng_actions, text="Salvar configuracoes", command=self.save_current_settings).pack(
            side="left"
        )
        ttk.Label(eng_actions, textvariable=self.engine_status_var, foreground="#374151").pack(
            side="left", padx=16
        )

        # =====================================================================
        # TAB 4: TREINO (Training Mode, Tolerance, Session Report & PGN)
        # =====================================================================
        tr_box = ttk.LabelFrame(tab_train, text="Treino de Lances com Feedback Tatico", padding=14)
        tr_box.pack(fill="x", pady=(0, 14))

        tr_row1 = ttk.Frame(tr_box)
        tr_row1.pack(fill="x", pady=4)
        ttk.Checkbutton(
            tr_row1,
            text="Ocultar melhores lances e setas no tabuleiro durante o treino",
            variable=self.training_mode_var,
            command=self._refresh_analysis_display,
        ).pack(side="left")

        ttk.Label(tr_row1, text="Tolerancia de avaliacao (centipawns):").pack(side="left", padx=(24, 6))
        sp_tol = ttk.Spinbox(
            tr_row1, from_=0, to=200, increment=10, textvariable=self.tolerance_cp_var, width=5
        )
        sp_tol.pack(side="left")
        self._add_tip(
            sp_tol,
            "Aceita qualquer lance entre as melhores linhas da engine que perca no maximo essa quantidade de centipawns.",
        )

        tr_row2 = ttk.Frame(tr_box)
        tr_row2.pack(fill="x", pady=(12, 4))
        ttk.Label(tr_row2, text="Digite seu lance (ex: Cf3, Nf3, Bc4, O-O, e2e4):").pack(side="left")
        guess_entry = tk.Entry(
            tr_row2,
            textvariable=self.guess_var,
            width=16,
            font=("Consolas", 11),
            bg="#ffffff",
            fg="#111827",
            insertbackground="#111827",
            relief="solid",
            bd=1,
        )
        guess_entry.pack(side="left", padx=8, ipady=3)
        guess_entry.bind("<Return>", lambda _e: self.check_guess())
        ttk.Button(tr_row2, text="Verificar lance", command=self.check_guess).pack(side="left", padx=4)
        ttk.Button(tr_row2, text="Revelar resposta", command=self.reveal_best_move).pack(side="left", padx=4)

        stats_box = ttk.LabelFrame(tab_train, text="Desempenho da Sessao e Exportacao", padding=14)
        stats_box.pack(fill="x")
        st_row = ttk.Frame(stats_box)
        st_row.pack(fill="x")
        ttk.Label(st_row, textvariable=self.score_label_var, font=("Segoe UI", 11, "bold")).pack(
            side="left"
        )
        ttk.Button(st_row, text="Zerar sessao", command=self.reset_training_session).pack(
            side="left", padx=14
        )
        ttk.Button(st_row, text="Exportar relatorio da sessao", command=self.export_session_report).pack(
            side="right", padx=4
        )
        ttk.Button(st_row, text="Salvar partida em PGN", command=self.export_pgn).pack(
            side="right", padx=4
        )
        ttk.Button(st_row, text="Limpar historico", command=self.clear_history).pack(
            side="right", padx=4
        )

        # ── Global Keyboard Shortcuts & Bottom Status Line ──
        self.bind("<F8>", lambda _e: self.full_pipeline())
        self.bind("<Alt-Left>", lambda _e: self.history_back())
        self.bind("<Alt-Right>", lambda _e: self.history_forward())

        status_footer = ttk.Frame(root_frame)
        status_footer.pack(fill="x", pady=(8, 0))
        ttk.Label(
            status_footer,
            textvariable=self.status,
            wraplength=1080,
            font=("Segoe UI", 10),
            foreground="#1f2937",
        ).pack(anchor="w")

    def _apply_loaded_settings(self) -> None:
        if self.settings.topmost:
            self.attributes("-topmost", True)
        if self.settings.region:
            try:
                self.region = parse_region(self.settings.region)
            except ValueError:
                self.region = None
        if self.settings.auto_monitor:
            self.after(800, self.toggle_monitoring)

    def collect_settings(self) -> AppSettings:
        return AppSettings(
            region=self.region_var.get().strip(),
            orientation=self.orientation_var.get(),
            engine_path=self.engine_var.get().strip(),
            depth=int(self.depth_var.get()),
            analysis_mode=self.analysis_mode_var.get(),
            movetime_sec=float(self.movetime_var.get()),
            engine_timeout_sec=self.settings.engine_timeout_sec,
            training_mode=bool(self.training_mode_var.get()),
            accept_tolerance_cp=int(self.tolerance_cp_var.get()),
            auto_orient=bool(self.auto_orient_var.get()),
            auto_turn=bool(self.auto_turn_var.get()),
            auto_monitor=bool(self.auto_monitor_var.get()),
            review_before_analysis=bool(self.review_before_analysis_var.get()),
            learn_from_corrections=self.settings.learn_from_corrections,
            topmost=bool(self.topmost_var.get()),
            window_geometry=self.geometry(),
        )

    def save_current_settings(self) -> None:
        try:
            self.settings = self.collect_settings()
            save_settings(self.settings, self.settings_path)
            self.status.set(f"Configuracoes salvas em {self.settings_path.name}.")
        except Exception as exc:
            self._set_status_indicator("error", "Falha ao salvar")
            self.status.set(f"Falha ao salvar configuracoes: {exc}")

    def _cancel_active_analysis(self, status_text: Optional[str] = None) -> None:
        """Cancels any in-flight engine analysis immediately so position changes never collide."""
        self._analysis_seq += 1
        self._is_analysing = False
        if self._stop_event is not None:
            self._stop_event.set()
            self._stop_event = None
        proc = self._active_engine_process
        self._active_engine_process = None
        if proc is not None:
            try:
                proc.terminate()  # type: ignore[attr-defined]
            except Exception:
                try:
                    proc.kill()  # type: ignore[attr-defined]
                except Exception:
                    pass
        if status_text is not None:
            self.engine_status_var.set(status_text)

    def _on_close(self) -> None:
        try:
            self._cancel_active_analysis()
            self.settings = self.collect_settings()
            save_settings(self.settings, self.settings_path)
        except Exception:
            pass
        self.destroy()

    def _toggle_topmost(self) -> None:
        self.attributes("-topmost", bool(self.topmost_var.get()))

    def _update_validation_badge(self) -> ValidationReport:
        fen = self.fen_var.get().strip()
        report = validate_fen(fen)
        if report.ok and not report.warnings:
            self.validation_var.set("Posicao valida")
            self.validation_badge.configure(bg="#dcfce7", fg="#166534")
        elif report.ok:
            self.validation_var.set(report.summary())
            self.validation_badge.configure(bg="#fef9c3", fg="#854d0e")
        else:
            self.validation_var.set(report.summary())
            self.validation_badge.configure(bg="#fee2e2", fg="#991b1b")
            self._set_status_indicator("error", "Posicao invalida")
        return report

    def _update_history_ui(self) -> None:
        total = len(self.history)
        idx = self.history.index + 1 if total else 0
        self.history_pos_var.set(f"Posicao {idx}/{total}")
        moves_str = self.history.recent_moves(count=10)
        self.recent_moves_var.set(
            f"Ultimos lances: {moves_str}" if moves_str else "Ultimos lances: (posicao inicial)"
        )
        self.btn_first.configure(state="normal" if self.history.can_back else "disabled")
        self.btn_back.configure(state="normal" if self.history.can_back else "disabled")
        self.btn_forward.configure(state="normal" if self.history.can_forward else "disabled")
        self.btn_last.configure(state="normal" if self.history.can_forward else "disabled")

    def _load_history_entry(self, entry: Optional[HistoryEntry]) -> None:
        if entry is None:
            return
        import chess

        self._cancel_active_analysis("Engine pronta")
        self.fen_var.set(entry.fen)
        try:
            b = chess.Board(entry.fen)
            self.turn_var.set("w" if b.turn == chess.WHITE else "b")
            self.prev_board_fen = b.board_fen()
        except Exception:
            pass
        self.last_board_image = None
        self.last_uncertain_squares = []
        self.last_detected_move = entry.move_uci
        self.last_lines = []
        self._render_engine_lines([])
        self._update_turn_banners()
        report = self._update_validation_badge()
        self._update_history_ui()
        self._update_preview(None, draw_arrows=False)
        self.status.set(
            f"Historico: posicao {self.history.index + 1}/{len(self.history)}"
            + (f" ({entry.move_label()})" if entry.move_label() else "")
        )
        if report.ok:
            self.analyse()

    def history_first(self) -> None:
        self._load_history_entry(self.history.first())

    def history_back(self) -> None:
        self._load_history_entry(self.history.back())

    def history_forward(self) -> None:
        self._load_history_entry(self.history.forward())

    def history_last(self) -> None:
        self._load_history_entry(self.history.last())

    def clear_history(self) -> None:
        self._cancel_active_analysis("Engine pronta")
        self.history.clear()
        self.prev_board_fen = None
        self.last_detected_move = None
        self.history.add(self.fen_var.get().strip(), source="atual")
        self._update_history_ui()
        self._update_turn_banners()
        self.status.set("Historico de posicoes reiniciado.")

    def apply_manual_fen(self) -> None:
        import chess

        self._cancel_active_analysis("Engine pronta")
        raw = self.fen_var.get().strip()
        try:
            clean = sanitize_fen(raw)
            b = chess.Board(clean)
            self.fen_var.set(clean)
            self.turn_var.set("w" if b.turn == chess.WHITE else "b")
            self.prev_board_fen = b.board_fen()
            self.last_recognition_confidence = 1.0
            self.last_board_image = None
            self.last_detected_move = None
            self.last_uncertain_squares = []
        except Exception as exc:
            self._update_validation_badge()
            self._set_status_indicator("error", "FEN invalida")
            self.status.set(f"FEN invalida: {exc}")
            return
        self.history.add(self.fen_var.get().strip(), source="manual")
        self._update_turn_banners()
        report = self._update_validation_badge()
        self._update_history_ui()
        self._update_preview(None, draw_arrows=False)
        if report.ok:
            self.analyse()
        else:
            self.status.set(report.summary())

    def _xy_to_inline_square(self, x: int, y: int, size: Optional[int] = None) -> Optional[str]:
        import chess

        board_px = size or self.BOARD_PX
        if not (0 <= x < board_px and 0 <= y < board_px):
            return None
        sq_px = board_px / 8.0
        col = int(x // sq_px)
        row = int(y // sq_px)
        orient = self.orientation_var.get()
        file_idx = col if orient == "white" else 7 - col
        rank_idx = 7 - row if orient == "white" else row
        return chess.square_name(chess.square(file_idx, rank_idx))

    def _apply_inline_edit(self, model: BoardEditModel) -> None:
        self._cancel_active_analysis("Posicao editada")
        new_fen = model.to_fen()
        changed = model.changed_squares()
        self.fen_var.set(new_fen)
        self.turn_var.set("w" if model.board.turn else "b")
        self.prev_board_fen = model.board.board_fen()
        self.history.replace_current(new_fen, source="editor")
        if changed and self.last_board_image is not None and self.settings.learn_from_corrections:
            learned = self.recognizer.learn_from_correction(
                self.last_board_image,
                new_fen,
                orientation=self.orientation_var.get(),
                squares=changed,
            )
            if learned:
                self.calib_badge.set(self._format_calib_badge())
        self.last_lines = []
        self._render_engine_lines([])
        self._update_turn_banners()
        self._update_validation_badge()
        self._update_history_ui()
        self.view_mode_var.set(True)
        self._update_preview(self.last_board_image, draw_arrows=False)
        self.status.set(
            f"Casa(s) ajustada(s): {', '.join(changed) if changed else 'posicao atualizada'}. Clique em 'Analisar posicao'."
        )

    def _on_inline_board_press(self, event) -> None:
        sq = self._xy_to_inline_square(event.x, event.y)
        if not sq:
            return
        tool = self.inline_tool_var.get()
        if tool == "move":
            self._inline_drag_from = sq
            return
        try:
            model = BoardEditModel(self.fen_var.get().strip())
        except Exception:
            return
        if tool == "erase":
            model.set_piece(sq, None)
        else:
            model.toggle_piece(sq, tool)
        self._apply_inline_edit(model)

    def _on_inline_board_release(self, event) -> None:
        if self.inline_tool_var.get() != "move" or not self._inline_drag_from:
            return
        from_sq = self._inline_drag_from
        self._inline_drag_from = None
        to_sq = self._xy_to_inline_square(event.x, event.y)
        if not to_sq or to_sq == from_sq:
            return
        try:
            model = BoardEditModel(self.fen_var.get().strip())
        except Exception:
            return
        if model.piece_at(from_sq) is None:
            return
        model.move_piece(from_sq, to_sq)
        self._apply_inline_edit(model)

    def _on_inline_board_right_click(self, event) -> None:
        sq = self._xy_to_inline_square(event.x, event.y)
        if not sq:
            return
        try:
            model = BoardEditModel(self.fen_var.get().strip())
        except Exception:
            return
        model.set_piece(sq, None)
        self._apply_inline_edit(model)

    def open_board_editor(self) -> None:
        self._cancel_active_analysis("Editando tabuleiro")
        if self._editor_dialog is not None:
            try:
                if self._editor_dialog.winfo_exists():
                    self._editor_dialog.destroy()
            except Exception:
                pass
            self._editor_dialog = None

        fen = self.fen_var.get().strip() or DEFAULT_START_FEN

        def on_editor_apply(
            new_fen: str,
            changed_squares: List[str],
            learn: bool,
            run_analysis: bool,
        ) -> None:
            import chess

            self._editor_dialog = None
            self.fen_var.set(new_fen)
            try:
                b = chess.Board(new_fen)
                self.turn_var.set("w" if b.turn == chess.WHITE else "b")
                self.prev_board_fen = b.board_fen()
                self.last_recognition_confidence = 1.0
            except Exception:
                pass
            self.history.replace_current(new_fen, source="editor")
            if learn and changed_squares and self.last_board_image is not None:
                learned = self.recognizer.learn_from_correction(
                    self.last_board_image,
                    new_fen,
                    orientation=self.orientation_var.get(),
                    squares=changed_squares,
                )
                if learned:
                    self.calib_badge.set(self._format_calib_badge())
            self.view_mode_var.set(True)
            self._update_turn_banners()
            self._update_validation_badge()
            self._update_history_ui()
            self._update_preview(self.last_board_image, draw_arrows=False)
            if run_analysis:
                self.analyse()
            else:
                self.status.set("Tabuleiro atualizado manualmente.")

        self._editor_dialog = BoardEditorDialog(
            self,
            fen=fen,
            orientation=self.orientation_var.get(),
            uncertain_squares=self.last_uncertain_squares,
            learn_default=self.settings.learn_from_corrections,
            on_apply=on_editor_apply,
        )

    def copy_fen(self) -> None:
        fen = self.fen_var.get().strip()
        if fen:
            self.clipboard_clear()
            self.clipboard_append(fen)
            self.status.set("FEN copiada para a area de transferencia.")

    def copy_analysis(self) -> None:
        fen = self.fen_var.get().strip()
        if not self.last_lines:
            self.copy_fen()
            return
        text = format_analysis_for_clipboard(fen, self.last_lines)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status.set("FEN, lances, variacao principal e avaliacao copiados para a area de transferencia.")

    def export_pgn(self) -> None:
        pgn = self.history.to_pgn()
        if not pgn.strip():
            self.status.set("Nenhum lance disponivel para exportar PGN.")
            return
        path = filedialog.asksaveasfilename(
            title="Salvar partida em PGN",
            defaultextension=".pgn",
            filetypes=[("Arquivos PGN", "*.pgn"), ("Todos os arquivos", "*.*")],
        )
        if path:
            Path(path).write_text(pgn, encoding="utf-8")
            self.status.set(f"PGN salvo em {path}")
        else:
            self.clipboard_clear()
            self.clipboard_append(pgn)
            self.status.set("PGN copiado para a area de transferencia.")

    def export_session_report(self) -> None:
        report_md = build_session_report(
            self.training_session,
            self.history,
            current_fen=self.fen_var.get().strip(),
            lines=self.last_lines,
        )
        path = filedialog.asksaveasfilename(
            title="Exportar Relatorio da Sessao",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Texto", "*.txt"), ("Todos os arquivos", "*.*")],
        )
        if path:
            Path(path).write_text(report_md, encoding="utf-8")
            self.status.set(f"Relatorio da sessao salvo em {path}")
        else:
            self.clipboard_clear()
            self.clipboard_append(report_md)
            self.status.set("Relatorio da sessao copiado para a area de transferencia.")

    def flip_turn(self) -> None:
        """Manually flips side to move ('w' <-> 'b'), updates FEN, and re-runs analysis."""
        import chess

        self._cancel_active_analysis("Engine pronta")
        new_turn = "b" if self.turn_var.get() == "w" else "w"
        self.turn_var.set(new_turn)
        try:
            b = chess.Board(self.fen_var.get().strip())
            b.turn = chess.WHITE if new_turn == "w" else chess.BLACK
            new_fen = sanitize_fen(b.fen())
            self.fen_var.set(new_fen)
            self.history.replace_current(new_fen, source="turno")
            self._update_turn_banners()
            report = self._update_validation_badge()
            self._update_history_ui()
            if report.ok:
                self.analyse()
        except Exception:
            pass

    def _update_turn_banners(self) -> None:
        my_color = "w" if self.orientation_var.get() == "white" else "b"
        active_turn = self.turn_var.get()
        turn_label = "Brancas" if active_turn == "w" else "Pretas"
        last_str = (
            f" • Ultimo lance: {self.last_detected_move[:2]} → {self.last_detected_move[2:4]}"
            if self.last_detected_move and len(self.last_detected_move) >= 4
            else ""
        )
        if active_turn == my_color:
            if not self._is_analysing:
                self._set_status_indicator("your_turn", f"{turn_label} jogam{last_str}")
            else:
                self.turn_banner_var.set(f"{turn_label} jogam{last_str}")
            self.lines_title_var.set(f"Melhores lances ({turn_label})")
        else:
            if not self._is_analysing:
                self._set_status_indicator("waiting", f"Vez das {turn_label}{last_str}")
            else:
                self.turn_banner_var.set(f"Vez das {turn_label}{last_str}")
            self.lines_title_var.set(f"Respostas esperadas ({turn_label})")

    def select_region_with_mouse(self) -> None:
        """Hides the main window briefly, grabs the screen, and opens the interactive snipping selector."""
        self.withdraw()
        self.after(180, self._launch_region_overlay)

    def _launch_region_overlay(self) -> None:
        try:
            screen_img = self.capture.capture()
            self.last_image = screen_img
        except Exception as exc:
            self.deiconify()
            self._set_status_indicator("error", "Falha na captura")
            self.status.set(f"Falha na captura: {exc}")
            return

        def on_selected(selected: Optional[BoardRegion]) -> None:
            self.deiconify()
            if self.topmost_var.get():
                self.attributes("-topmost", True)
            if selected is None:
                self.status.set("Selecao de area cancelada.")
                return
            self.region = selected
            self.region_var.set(f"{selected.x},{selected.y},{selected.width},{selected.height}")
            self.save_current_settings()
            try:
                self.last_board_image = self.capture.crop_region(self.last_image, selected)
                self.view_mode_var.set(False)
                self._update_preview(self.last_board_image, draw_arrows=False)
                self.notebook.select(0)
                self.status.set(
                    f"Area do tabuleiro definida ({self.region_var.get()}). Clique em 'Capturar e analisar (F8)'."
                )
            except Exception as exc:
                self._set_status_indicator("error", "Regiao invalida")
                self.status.set(f"Regiao invalida: {exc}")

        RegionSelectorOverlay(self, screen_img, on_selected)

    def flip_orientation(self) -> None:
        current = self.orientation_var.get()
        self.orientation_var.set("black" if current == "white" else "white")
        self._update_turn_banners()
        self._update_preview(self.last_board_image)

    def _draw_eval_bar(self, score_white_cp: Optional[int], mate_white: Optional[int]) -> None:
        import math

        height = self.BOARD_PX
        width = self.EVAL_BAR_W
        if mate_white is not None:
            white_ratio = 1.0 if mate_white > 0 else 0.0
        elif score_white_cp is None:
            white_ratio = 0.5
        else:
            clamped = max(-1000, min(1000, score_white_cp))
            white_ratio = 1.0 / (1.0 + math.exp(-0.004 * clamped))

        black_h = int(round((1.0 - white_ratio) * height))
        self.eval_bar_canvas.delete("all")
        self.eval_bar_canvas.create_rectangle(0, 0, width, black_h, fill="#262421", outline="")
        self.eval_bar_canvas.create_rectangle(0, black_h, width, height, fill="#f3f4f6", outline="")
        self.eval_bar_canvas.create_line(0, height // 2, width, height // 2, fill="#dc2626", width=2)

    def browse_engine(self) -> None:
        path = filedialog.askopenfilename(title="Selecionar executavel da engine UCI")
        if path:
            self.engine_var.set(path)
            self.save_current_settings()

    def _get_or_parse_region(self) -> BoardRegion:
        raw = self.region_var.get().strip()
        if raw:
            self.region = parse_region(raw)
            return self.region
        if self.region is not None:
            return self.region
        if self.last_image is None:
            self.last_image = self.capture.capture()
        self.region = self.detector.detect(self.last_image)
        self.region_var.set(f"{self.region.x},{self.region.y},{self.region.width},{self.region.height}")
        return self.region

    def _square_rect_px(self, square_name: str, size: Optional[int] = None) -> Tuple[int, int, int, int]:
        import chess

        board_px = size or self.BOARD_PX
        sq = chess.parse_square(square_name)
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        orientation = self.orientation_var.get()
        col = file_idx if orientation == "white" else 7 - file_idx
        row = 7 - rank_idx if orientation == "white" else rank_idx
        sq_size = board_px / 8.0
        x1 = int(round(col * sq_size))
        y1 = int(round(row * sq_size))
        x2 = int(round((col + 1) * sq_size))
        y2 = int(round((row + 1) * sq_size))
        return x1, y1, x2, y2

    def _square_center_px(self, square_name: str, size: Optional[int] = None) -> Tuple[int, int]:
        x1, y1, x2, y2 = self._square_rect_px(square_name, size)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _update_preview(self, board_img, draw_arrows: bool = True) -> None:
        try:
            import cv2
            from PIL import Image, ImageTk

            b_px = self.BOARD_PX
            use_diagram = bool(self.view_mode_var.get()) or board_img is None
            if use_diagram:
                canvas = render_board_bgr(
                    self.fen_var.get().strip() or DEFAULT_START_FEN,
                    orientation=self.orientation_var.get(),
                    size=b_px,
                    uncertain_squares=self.last_uncertain_squares,
                )
            else:
                canvas = cv2.resize(board_img, (b_px, b_px), interpolation=cv2.INTER_AREA)

            if self.last_detected_move and len(self.last_detected_move) >= 4:
                hl = canvas.copy()
                for sq_name in (self.last_detected_move[:2], self.last_detected_move[2:4]):
                    x1, y1, x2, y2 = self._square_rect_px(sq_name, b_px)
                    cv2.rectangle(hl, (x1, y1), (x2, y2), (35, 215, 245), -1)
                canvas = cv2.addWeighted(hl, 0.36, canvas, 0.64, 0)

            show_arrows = draw_arrows and bool(self.last_lines) and not self.training_mode_var.get()
            if show_arrows:
                overlay = canvas.copy()
                my_color = "w" if self.orientation_var.get() == "white" else "b"
                is_my_turn = self.turn_var.get() == my_color
                if is_my_turn:
                    arrow_styles = [
                        (0, (45, 180, 75), 9),    # 1st best move: crisp green BGR
                        (1, (220, 130, 35), 6),   # 2nd best move: crisp blue BGR
                    ]
                else:
                    arrow_styles = [
                        (0, (30, 125, 235), 7),   # Opponent's expected reply: orange BGR
                    ]
                for idx, color_bgr, thickness in reversed(arrow_styles):
                    if idx < len(self.last_lines):
                        uci = self.last_lines[idx].uci_move
                        if len(uci) >= 4:
                            p1 = self._square_center_px(uci[:2], b_px)
                            p2 = self._square_center_px(uci[2:4], b_px)
                            cv2.circle(overlay, p1, 13, color_bgr, -1, lineType=cv2.LINE_AA)
                            cv2.arrowedLine(
                                overlay, p1, p2, color_bgr, thickness, line_type=cv2.LINE_AA, tipLength=0.24
                            )
                canvas = cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0)

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self.preview_photo = ImageTk.PhotoImage(pil_img)
            self.board_canvas.delete("all")
            self.board_canvas.create_image(0, 0, image=self.preview_photo, anchor="nw")
        except Exception:
            pass

    def capture_screen(self) -> None:
        try:
            self.last_image = self.capture.capture()
            self.status.set("Tela capturada com sucesso.")
        except Exception as exc:
            self._set_status_indicator("error", "Falha na captura")
            self.status.set(f"Falha na captura: {exc}")

    def detect_board(self) -> None:
        if self.last_image is None:
            self.capture_screen()
        if self.last_image is None:
            return
        try:
            self.region = self.detector.detect(self.last_image)
            self.region_var.set(f"{self.region.x},{self.region.y},{self.region.width},{self.region.height}")
            self.last_board_image = self.capture.crop_region(self.last_image, self.region)
            self.view_mode_var.set(False)
            self._update_preview(self.last_board_image, draw_arrows=False)
            self.save_current_settings()
            self.status.set(f"Tabuleiro detectado em {self.region_var.get()}.")
        except Exception as exc:
            self._set_status_indicator("error", "Deteccao falhou")
            self.status.set(f"Deteccao automatica falhou: {exc}")

    def calibrate_from_screen(self) -> None:
        self._cancel_active_analysis("Engine pronta")
        if self.last_image is None:
            self.capture_screen()
        if self.last_image is None:
            return
        try:
            region = self._get_or_parse_region()
            board_img = self.capture.crop_region(self.last_image, region)
            self.last_board_image = board_img
            if self.auto_orient_var.get():
                det_orient = self.recognizer.detect_orientation(board_img)
                self.orientation_var.set(det_orient)
            self.view_mode_var.set(False)
            self.last_detected_move = None
            self.last_lines = []
            self._render_engine_lines([])
            self._update_preview(board_img, draw_arrows=False)
            count = self.recognizer.calibrate(
                board_img, fen=DEFAULT_START_FEN, orientation=self.orientation_var.get()
            )
            self.fen_var.set(DEFAULT_START_FEN)
            self.prev_board_fen = DEFAULT_START_BOARD_FEN
            self.calib_badge.set(self._format_calib_badge())
            self._set_status_indicator("ready", "Calibrado")
            self.status.set(
                f"Calibracao concluida ({count} tipos de peca salvos, {self.recognizer.theme_count} tema(s))."
            )
        except Exception as exc:
            self._set_status_indicator("error", "Falha na calibracao")
            self.status.set(f"Falha na calibracao: {exc}")

    def _apply_smart_recognition(self, board_img) -> RecognitionResult:
        import chess

        if self.auto_orient_var.get():
            det_orient = self.recognizer.detect_orientation(board_img)
            self.orientation_var.set(det_orient)

        orient = self.orientation_var.get()
        current_turn = self.turn_var.get()

        raw_res = self.recognizer.recognize(
            board_img,
            orientation=orient,
            side_to_move=current_turn,
        )

        if self.auto_turn_var.get():
            inferred_turn, last_mv = infer_turn_and_last_move(
                self.prev_board_fen,
                raw_res.board_fen,
                fallback_turn=current_turn,
                visual_turn=raw_res.inferred_turn,
                visual_last_move=raw_res.last_move_uci,
            )
            self.last_detected_move = last_mv
            b = chess.Board(raw_res.fen)
            b.turn = chess.WHITE if inferred_turn == "w" else chess.BLACK
            raw_res.fen = sanitize_fen(b.fen())
            try:
                b_clean = chess.Board(raw_res.fen)
                inferred_turn = "w" if b_clean.turn == chess.WHITE else "b"
            except Exception:
                pass
            self.turn_var.set(inferred_turn)
            raw_res.inferred_turn = inferred_turn
            raw_res.last_move_uci = self.last_detected_move
        else:
            raw_res.fen = sanitize_fen(raw_res.fen)
            try:
                b_clean = chess.Board(raw_res.fen)
                self.turn_var.set("w" if b_clean.turn == chess.WHITE else "b")
            except Exception:
                pass
            self.last_detected_move = raw_res.last_move_uci

        self.prev_board_fen = raw_res.board_fen
        self.last_uncertain_squares = list(raw_res.uncertain_squares)
        self.last_recognition_confidence = float(raw_res.confidence)
        self._update_turn_banners()
        return raw_res

    def recognize_board(self) -> Optional[RecognitionResult]:
        self._cancel_active_analysis("Reconhecendo...")
        if self.last_image is None:
            self.capture_screen()
        if self.last_image is None:
            return None
        try:
            region = self._get_or_parse_region()
            board_img = self.capture.crop_region(self.last_image, region)
            saved_prev_board = self.prev_board_fen
            result = self._apply_smart_recognition(board_img)
            report = validate_fen(result.fen)

            # If the board shifted on screen (e.g. switching from Play to Game Review / next game),
            # automatically re-detect the board coordinates and use them if they improve recognition.
            if not report.ok or result.confidence < 0.72:
                try:
                    auto_reg = self.detector.detect(self.last_image)
                    if (
                        abs(auto_reg.x - region.x) > 8
                        or abs(auto_reg.y - region.y) > 8
                        or abs(auto_reg.width - region.width) > 8
                    ):
                        alt_img = self.capture.crop_region(self.last_image, auto_reg)
                        self.prev_board_fen = saved_prev_board
                        alt_res = self._apply_smart_recognition(alt_img)
                        alt_rep = validate_fen(alt_res.fen)
                        if alt_rep.ok and (not report.ok or alt_res.confidence > result.confidence + 0.04):
                            self.region = auto_reg
                            self.region_var.set(
                                f"{auto_reg.x},{auto_reg.y},{auto_reg.width},{auto_reg.height}"
                            )
                            board_img = alt_img
                            result = alt_res
                        else:
                            # Restore original recognition state
                            self.prev_board_fen = result.board_fen
                            self.last_uncertain_squares = list(result.uncertain_squares)
                            self.last_recognition_confidence = float(result.confidence)
                            self.last_detected_move = result.last_move_uci
                            if result.inferred_turn in ("w", "b"):
                                self.turn_var.set(result.inferred_turn)
                            self._update_turn_banners()
                except Exception:
                    pass

            self.last_board_image = board_img
            self.last_lines = []
            self._render_engine_lines([])
            self.view_mode_var.set(False)
            self.fen_var.set(result.fen)
            self.history.add(result.fen, source="captura", move_uci_hint=result.last_move_uci)
            report = self._update_validation_badge()
            self._update_history_ui()
            self._update_preview(board_img, draw_arrows=False)

            pct = int(round(result.confidence * 100))
            if not report.ok:
                self.confidence_var.set(f"Reconhecimento: {pct}% (posicao invalida)")
                self._set_status_indicator("error", "Posicao invalida")
                self.status.set(f"{report.summary()} — Clique em 'Editar tabuleiro' para corrigir.")
            elif result.uncertain_squares:
                sq_list = ", ".join(result.uncertain_squares[:6])
                self.confidence_var.set(f"Reconhecimento: {pct}% (revisar: {sq_list})")
                self.status.set(
                    f"Posicao reconhecida ({pct}% confianca). Casas incertas: {sq_list}. "
                    "Arraste pecas no tabuleiro ou clique em 'Editar tabuleiro' se precisar corrigir."
                )
            else:
                self.confidence_var.set(f"Reconhecimento: {pct}%")
                self.status.set(f"Posicao reconhecida ({pct}% confianca).")
            return result
        except Exception as exc:
            self._set_status_indicator("error", "Erro no reconhecimento")
            self.status.set(f"Reconhecimento: {exc}")
            return None

    def full_pipeline(self) -> None:
        self._cancel_active_analysis()
        self.capture_screen()
        if self.last_image is None:
            return
        res = self.recognize_board()
        if res is None:
            return
        report = validate_fen(res.fen)
        if self.review_before_analysis_var.get():
            self.open_board_editor()
            return
        if not report.ok:
            self.view_mode_var.set(True)
            self._update_preview(self.last_board_image, draw_arrows=False)
            self._set_status_indicator("error", "Revisar posicao")
            self.status.set(
                f"{report.summary()} — Arraste pecas no tabuleiro para corrigir ou clique em 'Editar tabuleiro'."
            )
            return
        self.analyse()

    def toggle_monitoring(self) -> None:
        if self.auto_monitor_var.get():
            self._schedule_monitor()
        elif self.monitoring_job is not None:
            self.after_cancel(self.monitoring_job)
            self.monitoring_job = None

    def _schedule_monitor(self) -> None:
        if not self.auto_monitor_var.get():
            return
        try:
            img = self.capture.capture()
            self.last_image = img
            region = self._get_or_parse_region()
            board_img = self.capture.crop_region(img, region)
            old_board_fen = self.prev_board_fen or ""
            old_turn = self.turn_var.get()
            res = self._apply_smart_recognition(board_img)
            if res.board_fen != old_board_fen or self.turn_var.get() != old_turn:
                self._cancel_active_analysis()
                self.last_board_image = board_img
                self.fen_var.set(res.fen)
                report = self._update_validation_badge()
                suspicious = is_suspicious(report, res.confidence, len(res.uncertain_squares))
                self._update_preview(board_img, draw_arrows=False)
                if not suspicious:
                    self.history.add(res.fen, source="monitor", move_uci_hint=res.last_move_uci)
                    self._update_history_ui()
                    self.analyse()
                else:
                    self.status.set(f"Monitoramento: posicao incompleta ignorada ({report.summary()}).")
        except Exception as exc:
            self.status.set(f"Monitoramento: {exc}")
        self.monitoring_job = self.after(1500, self._schedule_monitor)

    def reset_training_session(self) -> None:
        self.training_session.reset()
        self.training_hits = 0
        self.training_misses = 0
        self.score_label_var.set(self.training_session.stats_text())
        self.status.set("Estatisticas da sessao de treino reiniciadas.")

    def check_guess(self) -> None:
        import chess

        guess_raw = self.guess_var.get().strip()
        if not guess_raw:
            self.status.set("Treino: digite seu lance (ex: Cf3, Nf3, Bc4 ou e2e4).")
            return
        if not self.last_lines:
            self.status.set("Treino: analise a posicao primeiro para comparar seu lance.")
            return
        fen = self.fen_var.get().strip()
        try:
            guess_uci = parse_user_move_to_uci(fen, guess_raw)
            board = chess.Board(fen)
            guess_san = board.san(chess.Move.from_uci(guess_uci))
        except ValueError as exc:
            self.status.set(f"Treino: {exc}")
            return

        white_to_move = board.turn == chess.WHITE
        tol = int(self.tolerance_cp_var.get())
        res = evaluate_guess(guess_uci, self.last_lines, white_to_move=white_to_move, tolerance_cp=tol)
        explanation = explain_move(fen, guess_uci)
        top = self.last_lines[0]

        self.training_session.record(
            TrainingAttempt(
                fen=fen,
                guess_uci=guess_uci,
                guess_san=guess_san,
                best_uci=top.uci_move,
                best_san=top.san_move,
                accepted=res.accepted,
                rank=res.rank,
                cp_loss=res.cp_loss,
                explanation=explanation,
            )
        )
        self.training_hits = self.training_session.hits
        self.training_misses = self.training_session.misses
        self.score_label_var.set(self.training_session.stats_text())

        if res.accepted:
            self.reveal_best_move()
            self.status.set(f"Correto ({guess_san}) — {res.message}! Motivo: {explanation}")
        else:
            piece_name = PT_PIECE_NAMES.get(top.piece_symbol.upper(), "peca")
            hint_sq = top.uci_move[:2]
            best_exp = explain_move(fen, top.uci_move)
            self.status.set(
                f"{res.message}. Seu lance ({guess_san}): {explanation}. "
                f"Dica: {piece_name} da casa {hint_sq} ({best_exp})."
            )

    def reveal_best_move(self) -> None:
        if not self.last_lines:
            self.status.set("Nenhuma analise disponivel ainda.")
            return
        self._render_engine_lines(self.last_lines, force_show=True)
        self._update_preview(self.last_board_image, draw_arrows=True)
        self.status.set(self.last_analysis_text)

    def _render_engine_lines(self, lines: List[EngineLine], force_show: bool = False) -> None:
        hide = self.training_mode_var.get() and not force_show
        if lines and not hide:
            self._draw_eval_bar(lines[0].score_white_cp, lines[0].mate_white)
        else:
            self._draw_eval_bar(0, None)

        fen = self.fen_var.get().strip()
        for idx, row_widgets in enumerate(self.line_rows):
            card: tk.Frame = row_widgets["card"]  # type: ignore[assignment]
            eval_lbl: tk.Label = row_widgets["eval"]  # type: ignore[assignment]
            icon_lbl: tk.Label = row_widgets["icon"]  # type: ignore[assignment]
            move_lbl: tk.Label = row_widgets["move"]  # type: ignore[assignment]
            conf_lbl: tk.Label = row_widgets["conf"]  # type: ignore[assignment]
            pv_lbl: tk.Label = row_widgets["pv"]  # type: ignore[assignment]
            exp_lbl: tk.Label = row_widgets["exp"]  # type: ignore[assignment]

            if hide:
                card.configure(highlightbackground="#d1d5db")
                eval_lbl.configure(text=" ? ", bg="#e5e7eb", fg="#374151")
                icon_lbl.configure(image="")
                move_lbl.configure(text="Lance oculto" if idx == 0 else "-")
                conf_lbl.configure(text="")
                pv_lbl.configure(
                    text="Modo treino ativo — digite seu palpite abaixo ou clique em 'Revelar resposta'."
                    if idx == 0
                    else ""
                )
                exp_lbl.configure(text="")
                continue

            if idx < len(lines):
                line = lines[idx]
                white_ahead = (
                    (line.mate_white is not None and line.mate_white > 0)
                    or (line.score_white_cp is not None and line.score_white_cp >= 0)
                )
                # Consistent colors for White vs Black advantage
                bg_col = "#f8f9fa" if white_ahead else "#1f2937"
                fg_col = "#111827" if white_ahead else "#f9fafb"
                card_border = "#16a34a" if idx == 0 else "#d1d5db"
                card.configure(highlightbackground=card_border)
                eval_lbl.configure(text=f" {line.eval_text} ", bg=bg_col, fg=fg_col)

                icon = self.piece_icons.get(line.piece_symbol)
                if icon is not None:
                    icon_lbl.configure(image=icon)
                else:
                    icon_lbl.configure(image="")

                origin, dest = line.uci_move[:2], line.uci_move[2:4]
                move_lbl.configure(text=f"{line.figurine_move}  ({origin} → {dest})")

                conf_pct = compute_line_confidence(idx, lines, self.last_recognition_confidence)
                conf_lbl.configure(text=f"Confianca: {conf_pct}%")
                pv_lbl.configure(text=f"Variacao principal: {line.pv_figurine}")
                exp_lbl.configure(text=explain_move(fen, line.uci_move))
            else:
                card.configure(highlightbackground="#e5e7eb")
                eval_lbl.configure(text=" - ", bg="#f3f4f6", fg="#6b7280")
                icon_lbl.configure(image="")
                if idx == 0 and self._is_analysing:
                    move_lbl.configure(text="Calculando melhores lances...")
                    pv_lbl.configure(text="Aguarde a resposta do Stockfish...")
                elif idx == 0:
                    move_lbl.configure(text="Pronto para analisar")
                    pv_lbl.configure(text="Use 'Capturar e analisar (F8)' para ver os melhores lances.")
                else:
                    move_lbl.configure(text="-")
                    pv_lbl.configure(text="")
                conf_lbl.configure(text="")
                exp_lbl.configure(text="")

    def _refresh_analysis_display(self) -> None:
        self._render_engine_lines(self.last_lines)
        self._update_preview(self.last_board_image, draw_arrows=True)

    def _on_analysis_success(self, payload: object) -> None:
        import chess

        if isinstance(payload, tuple) and len(payload) == 2:
            fen, lines = payload  # type: ignore[misc]
        else:
            fen = self.fen_var.get().strip()
            lines = payload  # type: ignore[assignment]

        self._is_analysing = False
        self.last_lines = list(lines)

        if not self.last_lines:
            self.last_best_move = None
            try:
                b = chess.Board(fen)
                if b.is_checkmate():
                    winner = "Pretas venceram" if b.turn == chess.WHITE else "Brancas venceram"
                    end_msg = f"Fim de jogo: Xeque-mate ({winner})."
                elif b.is_game_over():
                    end_msg = "Fim de jogo: Empate (afogamento ou material insuficiente)."
                else:
                    end_msg = "Nenhum lance legal disponivel nesta posicao."
            except Exception:
                end_msg = "Nenhum lance legal disponivel nesta posicao."
            self.last_analysis_text = end_msg
            self.engine_status_var.set("Fim de jogo")
            self._set_status_indicator("ready", "Fim de jogo")
            self._render_engine_lines([])
            self._update_preview(self.last_board_image, draw_arrows=False)
            self.status.set(end_msg)
            return

        top = self.last_lines[0]
        self.last_best_move = top.uci_move
        piece_name = PT_PIECE_NAMES.get(top.piece_symbol.upper(), "Peca")
        reason = explain_move(fen, top.uci_move)
        depth_str = f" (prof. {top.depth})" if top.depth else ""
        self.last_analysis_text = (
            f"Melhor lance: {top.figurine_move} ({piece_name} {top.uci_move[:2]} → {top.uci_move[2:4]}) "
            f"| Avaliacao: {top.eval_text}{depth_str} | {reason}"
        )
        self.history.set_evaluation(fen, top.eval_text, top.san_move)
        self.engine_status_var.set(f"Concluido{depth_str}")
        self._update_turn_banners()
        self._render_engine_lines(self.last_lines)
        self._update_preview(self.last_board_image, draw_arrows=True)

        if self.training_mode_var.get():
            msg = "Posicao analisada. Modo treino ativo: digite seu lance para verificar."
        else:
            msg = self.last_analysis_text
        self.status.set(msg)

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                req_id, kind, payload = self._ui_queue.get_nowait()
                if req_id != self._analysis_seq:
                    continue
                if kind == "progress":
                    self.engine_status_var.set(str(payload))
                elif kind == "ok":
                    self._on_analysis_success(payload)
                elif kind == "error":
                    self._is_analysing = False
                    self._set_status_indicator("error", "Falha na analise")
                    self.engine_status_var.set("Aviso na engine")
                    self.status.set(f"Aviso na analise: {payload}")
        except queue.Empty:
            pass
        try:
            if self.winfo_exists():
                self.after(50, self._poll_ui_queue)
        except tk.TclError:
            pass

    def stop_analysis(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._is_analysing:
            self.engine_status_var.set("Interrompendo...")
        else:
            self.engine_status_var.set("Engine pronta")

    def analyse(self) -> None:
        try:
            fen = self.fen_var.get().strip()
            depth = int(self.depth_var.get())
            mode = self.analysis_mode_var.get()
            movetime_sec = float(self.movetime_var.get())
            engine_path = self.engine_var.get().strip()
        except Exception as exc:
            self._set_status_indicator("error", "Parametros invalidos")
            self.status.set(f"Parametros invalidos: {exc}")
            return

        try:
            import chess

            fen = sanitize_fen(fen)
            b = chess.Board(fen)
            self.fen_var.set(fen)
            self.turn_var.set("w" if b.turn == chess.WHITE else "b")
        except Exception:
            pass

        report = self._update_validation_badge()
        if not report.ok:
            self._cancel_active_analysis("Posicao invalida")
            self._set_status_indicator("error", "Posicao invalida")
            self.status.set(f"{report.summary()} — Clique em 'Editar tabuleiro' para corrigir antes de analisar.")
            return

        # Cancel any previous engine process before starting a new search
        self._cancel_active_analysis()

        stop_event = threading.Event()
        self._stop_event = stop_event

        self._analysis_seq += 1
        req_id = self._analysis_seq
        self._is_analysing = True
        self.last_lines = []
        ui_queue = self._ui_queue
        movetime_ms = int(round(max(0.2, movetime_sec) * 1000)) if mode == "time" else None
        mode_desc = f"{movetime_sec:.1f}s" if mode == "time" else f"prof. {depth}"
        self._set_status_indicator("analysing")
        self._update_turn_banners()
        self.engine_status_var.set(f"Calculando ({mode_desc})...")
        self._render_engine_lines([])
        self.status.set(f"Analisando posicao ({mode_desc})...")

        def on_progress(text: str) -> None:
            ui_queue.put((req_id, "progress", text))

        def on_process(proc: object) -> None:
            if req_id == self._analysis_seq:
                self._active_engine_process = proc

        def work() -> None:
            try:
                lines = UciEngine(
                    engine_path or find_default_engine(),
                    timeout=max(15.0, movetime_sec + 5.0),
                ).analyse_lines(
                    fen,
                    depth=depth,
                    multipv=3,
                    movetime_ms=movetime_ms,
                    on_progress=on_progress,
                    stop_event=stop_event,
                    on_process=on_process,
                )
                ui_queue.put((req_id, "ok", (fen, lines)))
            except Exception as exc:
                if not stop_event.is_set():
                    ui_queue.put((req_id, "error", str(exc)))

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess Coach: captura de tela, visao computacional e analise UCI")
    parser.add_argument(
        "--engine",
        default="",
        help="caminho para uma engine UCI, como Stockfish ou OpenTal (auto-detectado por padrao)",
    )
    args = parser.parse_args()
    ChessCoachApp(args.engine).mainloop()


if __name__ == "__main__":
    main()
