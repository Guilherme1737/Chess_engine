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
from typing import Dict, List, Optional, Tuple

DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEFAULT_START_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration_data.npz"
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
    4. Exact legal move transition from the previous captured board state.
    """
    import chess

    # 1. Initial position is always White to move
    if curr_board_fen == DEFAULT_START_BOARD_FEN:
        return "w", None

    curr_base = chess.BaseBoard(curr_board_fen)

    # 2. If a king is currently in check, it MUST be that king's turn to move
    white_king_sq = curr_base.king(chess.WHITE)
    black_king_sq = curr_base.king(chess.BLACK)
    if white_king_sq is not None and curr_base.is_attacked_by(chess.BLACK, white_king_sq):
        inferred_by_check = "w"
    elif black_king_sq is not None and curr_base.is_attacked_by(chess.WHITE, black_king_sq):
        inferred_by_check = "b"
    else:
        inferred_by_check = None

    # 3. If visual highlight detection found a clear last move on the board (from_sq empty -> to_sq occupied)
    if visual_turn in ("w", "b") and visual_last_move:
        turn_out = inferred_by_check if inferred_by_check is not None else visual_turn
        return turn_out, visual_last_move

    # 4. Compare against previous board position if available and different
    if prev_board_fen and prev_board_fen != curr_board_fen:
        for candidate_turn_color, next_turn_code in ((chess.WHITE, "b"), (chess.BLACK, "w")):
            try:
                test_board = chess.Board(
                    f"{prev_board_fen} {'w' if candidate_turn_color == chess.WHITE else 'b'} KQkq - 0 1"
                )
                for mv in test_board.legal_moves:
                    test_board.push(mv)
                    if test_board.board_fen() == curr_board_fen:
                        return (inferred_by_check or next_turn_code), mv.uci()
                    test_board.pop()
            except Exception:
                pass

        # Fallback: inspect which color vacated a square and arrived on a new square
        try:
            prev_base = chess.BaseBoard(prev_board_fen)
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

    if inferred_by_check is not None:
        return inferred_by_check, visual_last_move

    return fallback_turn, visual_last_move


def build_engine_line(
    fen: str,
    multipv: int,
    pv_uci: List[str],
    score_turn_cp: Optional[int] = None,
    mate_turn: Optional[int] = None,
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
    sim_board = board.copy()
    for idx, uci_str in enumerate(pv_uci[:6]):
        try:
            mv = chess.Move.from_uci(uci_str)
            if mv not in sim_board.legal_moves:
                break
            w_turn = sim_board.turn == chess.WHITE
            mv_san = sim_board.san(mv)
            mv_fig = san_to_figurine(mv_san, w_turn)
            if w_turn:
                pv_parts.append(f"{sim_board.fullmove_number}. {mv_fig}")
            elif idx == 0:
                pv_parts.append(f"{sim_board.fullmove_number}... {mv_fig}")
            else:
                pv_parts.append(mv_fig)
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
            break

    try:
        return board.parse_san(raw).uci()
    except ValueError:
        pass

    pt_to_en = {"C": "N", "B": "B", "T": "R", "D": "Q", "R": "K"}
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

    def save_calibration(self, path: Path) -> None:
        import numpy as np

        if not self.calibrated or self.bg_light is None or self.bg_dark is None:
            return
        banks = self.template_banks if self.template_banks else ([self.templates] if self.templates else [])
        if not banks:
            return
        latest = banks[-1]
        arrays: Dict[str, object] = {
            "bg_light": self.bg_light,
            "bg_dark": self.bg_dark,
            "symbols": np.array(list(latest.keys())),
            "bank_count": np.array(len(banks), dtype=np.int32),
        }
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
        np.savez_compressed(path, **arrays)

    def load_calibration(self, path: Path) -> None:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            self.bg_light = data["bg_light"].astype(np.float32)
            self.bg_dark = data["bg_dark"].astype(np.float32)
            self.template_banks = []
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

                if self.calibrated and active_bank:
                    best_sym = None
                    best_score = -1.0
                    second_score = -1.0
                    same_color_pool = {sym: tpl for sym, tpl in active_bank.items() if sym.isupper() == is_white}
                    pool = same_color_pool if same_color_pool else active_bank

                    for sym, tpl in pool.items():
                        s_gray = self._similarity(feats["neutral_gray"], tpl["neutral_gray"])
                        s_mask = self._similarity(feats["fg_mask"], tpl["fg_mask"])
                        s_iou = self._mask_iou(feats["fg_mask"], tpl["fg_mask"])
                        s_edge = self._similarity(feats["edges"], tpl["edges"])
                        score = 0.45 * s_gray + 0.25 * s_mask + 0.15 * s_iou + 0.15 * s_edge
                        if score > best_score:
                            second_score = best_score
                            best_score = score
                            best_sym = sym
                        elif score > second_score:
                            second_score = score

                    margin = max(0.0, best_score - second_score)
                    conf = float(np.clip(0.65 * max(0.0, best_score) + 0.35 * min(1.0, margin * 3.0), 0.0, 1.0))
                    grid[r][c] = best_sym
                    confidences.append(conf)
                    if conf < 0.55:
                        uncertain_squares.append(sq_name)
                else:
                    best_type = "P"
                    best_score = -1.0
                    for p_type, syn_mask in self._synthetic_masks.items():
                        score = 0.65 * self._similarity(feats["fg_mask"], syn_mask) + 0.35 * self._mask_iou(
                            feats["fg_mask"], syn_mask
                        )
                        if score > best_score:
                            best_score = score
                            best_type = p_type
                    sym = best_type if is_white else best_type.lower()
                    conf = float(np.clip(best_score * 0.80, 0.25, 0.80))
                    grid[r][c] = sym
                    confidences.append(conf)
                    if conf < 0.55:
                        uncertain_squares.append(sq_name)

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


class LocalChessEvaluator:
    """
    Tactical & positional alpha-beta evaluator with Piece-Square Tables and Quiescence Search
    using python-chess when no external UCI binary is configured.
    """

    PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}

    # Piece-Square Tables (from White's perspective, a1..h8, rank 1 at bottom of table)
    PST = {
        1: (  # Pawns
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10,-20,-20, 10, 10,  5,
            5, -5,-10,  0,  0,-10, -5,  5,
            0,  0,  0, 22, 22,  0,  0,  0,
            5,  5, 10, 25, 25, 10,  5,  5,
           10, 10, 20, 30, 30, 20, 10, 10,
           50, 50, 50, 50, 50, 50, 50, 50,
            0,  0,  0,  0,  0,  0,  0,  0,
        ),
        2: (  # Knights
           -50,-35,-30,-30,-30,-30,-35,-50,
           -35,-20,  0,  5,  5,  0,-20,-35,
           -30,  5, 15, 20, 20, 15,  5,-30,
           -30,  5, 20, 25, 25, 20,  5,-30,
           -30,  5, 15, 25, 25, 15,  5,-30,
           -30,  5, 15, 15, 15, 15,  5,-30,
           -35,-20,  0,  0,  0,  0,-20,-35,
           -50,-35,-30,-30,-30,-30,-35,-50,
        ),
        3: (  # Bishops
           -20,-10,-10,-10,-10,-10,-10,-20,
           -10,  5,  0,  0,  0,  0,  5,-10,
           -10, 10, 10, 15, 15, 10, 10,-10,
           -10,  0, 15, 15, 15, 15,  0,-10,
           -10,  5, 10, 15, 15, 10,  5,-10,
           -10,  0, 10, 10, 10, 10,  0,-10,
           -10,  0,  0,  0,  0,  0,  0,-10,
           -20,-10,-10,-10,-10,-10,-10,-20,
        ),
        4: (  # Rooks
             0,  0,  0,  5,  5,  0,  0,  0,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             5, 10, 10, 10, 10, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0,
        ),
        5: (  # Queens
           -20,-10,-10, -5, -5,-10,-10,-20,
           -10,  0,  0,  0,  0,  0,  0,-10,
           -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
             0,  0,  5,  5,  5,  5,  0, -5,
           -10,  5,  5,  5,  5,  5,  0,-10,
           -10,  0,  5,  0,  0,  0,  0,-10,
           -20,-10,-10, -5, -5,-10,-10,-20,
        ),
        6: (  # King
            20, 30, 10,  0,  0, 10, 30, 20,
            20, 20,  0,  0,  0,  0, 20, 20,
           -10,-20,-20,-20,-20,-20,-20,-10,
           -20,-30,-30,-40,-40,-30,-30,-20,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
        ),
    }

    @classmethod
    def evaluate_board(cls, board, depth_remaining: int = 0) -> int:
        import chess

        if board.is_checkmate():
            return -99999 - depth_remaining
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0
        score = 0
        for sq, piece in board.piece_map().items():
            val = cls.PIECE_VALUES.get(piece.piece_type, 0)
            pst_table = cls.PST.get(piece.piece_type)
            if pst_table is not None:
                idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
                bonus = pst_table[idx]
            else:
                bonus = 0
            total = val + bonus
            score += total if piece.color == chess.WHITE else -total
        return score if board.turn == chess.WHITE else -score

    @classmethod
    def _move_order_key(cls, board, move) -> int:
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v_val = cls.PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
            a_val = cls.PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            return 10000 + 10 * v_val - a_val
        if move.promotion:
            return 9000
        return 0

    @classmethod
    def _quiesce(cls, board, alpha: int, beta: int, qdepth: int = 6) -> int:
        if board.is_checkmate():
            return -99999 - qdepth
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0

        in_check = board.is_check()
        if not in_check:
            stand_pat = cls.evaluate_board(board, 0)
            if qdepth <= 0:
                return stand_pat
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            candidate_moves = list(board.generate_legal_captures())
        else:
            # When in check, stand-pat is illegal and all legal check evasions (including non-captures) must be searched
            if qdepth <= 0:
                return cls.evaluate_board(board, 0)
            candidate_moves = list(board.legal_moves)

        candidate_moves.sort(key=lambda mv: cls._move_order_key(board, mv), reverse=True)
        for mv in candidate_moves:
            board.push(mv)
            score = -cls._quiesce(board, -beta, -alpha, qdepth - 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    @classmethod
    def analyse_lines(
        cls,
        fen: str,
        depth: int = 3,
        multipv: int = 3,
        max_time_sec: Optional[float] = 4.0,
    ) -> List[EngineLine]:
        import chess

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise RuntimeError("A posicao informada nao possui lances legais")

        target_depth = max(1, int(depth))
        deadline = (time.monotonic() + max_time_sec) if max_time_sec is not None else None

        def negamax(b, d: int, alpha: int, beta: int) -> Tuple[int, List[str]]:
            if b.is_game_over():
                return cls.evaluate_board(b, d), []
            if d <= 0:
                return cls._quiesce(b, alpha, beta, qdepth=6), []
            max_eval = -1000000
            best_pv: List[str] = []
            ordered = list(b.legal_moves)
            ordered.sort(key=lambda m: cls._move_order_key(b, m), reverse=True)
            for mv in ordered:
                if deadline is not None and time.monotonic() > deadline and d > 1:
                    break
                b.push(mv)
                child_eval, child_pv = negamax(b, d - 1, -beta, -alpha)
                val = -child_eval
                b.pop()
                if val > max_eval:
                    max_eval = val
                    best_pv = [mv.uci()] + child_pv
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    break
            if not best_pv and ordered:
                return cls.evaluate_board(b, d), []
            return max_eval, best_pv

        # Iterative deepening from depth 1 up to target_depth so requested depth is genuinely respected
        ordered_root = list(legal_moves)
        ordered_root.sort(key=lambda m: cls._move_order_key(board, m), reverse=True)
        scored_moves: List[Tuple[int, List[str]]] = []

        for current_depth in range(1, target_depth + 1):
            if deadline is not None and current_depth > 2 and time.monotonic() > deadline:
                break
            iteration_scores: List[Tuple[int, List[str], object]] = []
            aborted = False
            for move in ordered_root:
                if deadline is not None and current_depth > 2 and time.monotonic() > deadline:
                    aborted = True
                    break
                board.push(move)
                child_eval, child_pv = negamax(board, current_depth - 1, -1000000, 1000000)
                score = -child_eval
                board.pop()
                iteration_scores.append((score, [move.uci()] + child_pv, move))

            if not aborted or not scored_moves:
                iteration_scores.sort(key=lambda item: item[0], reverse=True)
                scored_moves = [(sc, pv) for sc, pv, _ in iteration_scores]
                ordered_root = [mv for _, _, mv in iteration_scores]
            if aborted:
                break

        lines: List[EngineLine] = []
        for idx, (score, pv) in enumerate(scored_moves[: max(1, multipv)], start=1):
            is_mate = abs(score) >= 90000
            mate_moves = max(1, (len(pv) + 1) // 2)
            mate_turn = (mate_moves if score > 0 else -mate_moves) if is_mate else None
            cp_turn = None if is_mate else score
            lines.append(
                build_engine_line(
                    fen=fen,
                    multipv=idx,
                    pv_uci=pv,
                    score_turn_cp=cp_turn,
                    mate_turn=mate_turn,
                )
            )
        return lines

    @classmethod
    def analyse(cls, fen: str, depth: int = 3) -> Tuple[str, str]:
        lines = cls.analyse_lines(fen, depth=depth, multipv=1)
        top = lines[0]
        return top.uci_move, f"Melhor lance: {top.figurine_move} ({top.uci_move}) | avaliacao: {top.eval_text} (avaliador local)"


class UciEngine:
    def __init__(self, executable: str, timeout: float = 15.0):
        self.executable = executable
        self.timeout = timeout

    def analyse_lines(self, fen: str, depth: int, multipv: int = 3) -> List[EngineLine]:
        if not self.executable:
            raise RuntimeError("Informe o caminho de uma engine UCI, por exemplo OpenTal ou Stockfish")
        executable = self.executable if os.path.isfile(self.executable) else shutil.which(self.executable)
        if not executable or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Engine nao executavel: {self.executable}")
        commands = [
            "uci",
            f"setoption name MultiPV value {max(1, multipv)}",
            "isready",
            f"position fen {fen}",
            f"go depth {depth}",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        raw_lines: Dict[int, Tuple[Optional[int], Optional[int], List[str]]] = {}
        best_move: Optional[str] = None

        with subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        ) as process:
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                for command in commands:
                    process.stdin.write(command + "\n")
                process.stdin.flush()
            except OSError as exc:
                process.kill()
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

            deadline = time.monotonic() + self.timeout
            timed_out = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    item = line_queue.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    if process.poll() is not None and line_queue.empty():
                        break
                    continue

                if item is None:
                    break
                line = item.strip()
                if line.startswith("info") and " pv " in f" {line} ":
                    tokens = line.split()
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
                    pv_idx = tokens.index("pv")
                    pv_moves = tokens[pv_idx + 1 :]
                    if pv_moves:
                        raw_lines[mpv] = (cp_val, mate_val, pv_moves)
                elif line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != "(none)":
                        best_move = parts[1]
                    break

            try:
                if process.poll() is None:
                    if timed_out:
                        process.kill()
                    else:
                        process.stdin.write("quit\n")
                        process.stdin.flush()
                process.wait(timeout=2)
            except Exception:
                process.kill()
                process.wait(timeout=2)

            if timed_out:
                raise TimeoutError(f"A engine UCI excedeu o tempo limite de {self.timeout:.1f}s sem retornar bestmove")
            if not best_move:
                raise RuntimeError("A engine nao retornou bestmove")

        if not raw_lines:
            raw_lines[1] = (None, None, [best_move])

        results: List[EngineLine] = []
        for mpv in sorted(raw_lines.keys())[: max(1, multipv)]:
            cp_val, mate_val, pv_moves = raw_lines[mpv]
            try:
                results.append(
                    build_engine_line(
                        fen=fen,
                        multipv=mpv,
                        pv_uci=pv_moves,
                        score_turn_cp=cp_val,
                        mate_turn=mate_val,
                    )
                )
            except Exception:
                continue

        if not results:
            results.append(build_engine_line(fen=fen, multipv=1, pv_uci=[best_move]))
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


class ChessCoachApp(tk.Tk):
    def __init__(self, engine_path: str = ""):
        super().__init__()
        self.title("Chess Coach")
        self.geometry("1040x760")
        self.capture = ScreenCapture()
        self.detector = BoardDetector()
        self.recognizer = PieceRecognizer()
        self.last_image = None
        self.last_board_image = None
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
        self._ui_queue: queue.Queue[Tuple[int, str, object]] = queue.Queue()
        self._init_piece_icons()
        self._build_ui(engine_path or find_default_engine())
        self.after(50, self._poll_ui_queue)

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

    def _build_ui(self, engine_path: str) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)

        header = ttk.Frame(frame)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Chess Coach", font=("TkDefaultFont", 20, "bold")).pack(side="left")

        self.topmost_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            header,
            text="Fixar janela no topo (Always on Top)",
            variable=self.topmost_var,
            command=self._toggle_topmost,
        ).pack(side="right", padx=(14, 0))

        self.calib_badge = tk.StringVar(
            value="Calibrado" if self.recognizer.calibrated else "Nao calibrado (usando silhuetas padrao)"
        )
        ttk.Label(header, textvariable=self.calib_badge).pack(side="right")

        # Turn / Move State Banner at top of window so the user immediately sees if it's their turn or they already played
        self.turn_state_banner = tk.Label(
            frame,
            text="Pronto para capturar o tabuleiro (Pressione F8 ou clique em Capturar + Reconhecer + Analisar)",
            font=("Segoe UI Symbol", 11, "bold"),
            bg="#e8f4ea",
            fg="#1b5e20",
            padx=10,
            pady=6,
            anchor="w",
        )
        self.turn_state_banner.pack(fill="x", pady=(0, 10))

        main_pane = ttk.Frame(frame)
        main_pane.pack(fill="both", expand=True)

        left_col = ttk.Frame(main_pane)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 14))

        right_col = ttk.LabelFrame(main_pane, text="Tabuleiro & Barra de Avaliacao", padding=10)
        right_col.pack(side="right", fill="y")

        preview_container = ttk.Frame(right_col)
        preview_container.pack(padx=4, pady=4)

        self.eval_bar_canvas = tk.Canvas(preview_container, width=22, height=240, bg="#262421", highlightthickness=1)
        self.eval_bar_canvas.pack(side="left", padx=(0, 6))
        self._draw_eval_bar(0, None)

        self.preview_label = ttk.Label(
            preview_container, text="Nenhum tabuleiro\ncapturado ainda", anchor="center", width=30
        )
        self.preview_label.pack(side="left")

        self.confidence_var = tk.StringVar(value="Confianca: -")
        ttk.Label(right_col, textvariable=self.confidence_var, wraplength=260).pack(anchor="w", pady=(6, 2))
        self.turn_banner_var = tk.StringVar(value="Vez: Brancas (w)")
        ttk.Label(right_col, textvariable=self.turn_banner_var, font=("Segoe UI Symbol", 9, "bold"), wraplength=260).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            right_col,
            text="Setas: Verde = 1º lance | Azul = 2º lance\nAmarelo = ultimo lance detectado na tela",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w")

        # Capture & Board Region controls
        controls = ttk.Frame(left_col)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Button(controls, text="Capturar tela", command=self.capture_screen).pack(side="left")
        ttk.Button(controls, text="Detectar auto", command=self.detect_board).pack(side="left", padx=4)
        ttk.Button(controls, text="Selecionar area com mouse ✂", command=self.select_region_with_mouse).pack(
            side="left", padx=4
        )
        ttk.Button(controls, text="Reconhecer pecas", command=self.recognize_board).pack(side="left", padx=4)
        ttk.Button(controls, text="Calibrar posicao inicial", command=self.calibrate_from_screen).pack(side="left")

        region_row = ttk.Frame(left_col)
        region_row.pack(fill="x", pady=3)
        ttk.Label(region_row, text="Regiao x,y,w,h:").pack(side="left")
        self.region_var = tk.StringVar()
        ttk.Entry(region_row, textvariable=self.region_var, width=16).pack(side="left", padx=4)
        ttk.Label(region_row, text="Eu jogo de:").pack(side="left", padx=(8, 2))
        self.orientation_var = tk.StringVar(value="white")
        ttk.Combobox(
            region_row,
            textvariable=self.orientation_var,
            values=("white", "black"),
            state="readonly",
            width=6,
        ).pack(side="left")
        ttk.Button(region_row, text="⇅", command=self.flip_orientation, width=3).pack(side="left", padx=2)
        ttk.Label(region_row, text="Vez atual:").pack(side="left", padx=(8, 2))
        self.turn_var = tk.StringVar(value="w")
        ttk.Combobox(
            region_row,
            textvariable=self.turn_var,
            values=("w", "b"),
            state="readonly",
            width=3,
        ).pack(side="left")
        ttk.Button(region_row, text="Trocar Vez (w⇄b)", command=self.flip_turn).pack(side="left", padx=4)

        auto_row = ttk.Frame(left_col)
        auto_row.pack(fill="x", pady=2)
        self.auto_orient_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_row, text="Auto-detectar orientacao", variable=self.auto_orient_var).pack(
            side="left", padx=(0, 12)
        )
        self.auto_turn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            auto_row,
            text="Auto-detectar vez pelo destaque do ultimo lance na tela",
            variable=self.auto_turn_var,
        ).pack(side="left")

        # FEN input/output with Copy button
        ttk.Label(left_col, text="FEN da posicao:").pack(anchor="w", pady=(6, 2))
        fen_row = ttk.Frame(left_col)
        fen_row.pack(fill="x")
        self.fen_var = tk.StringVar(value=DEFAULT_START_FEN)
        ttk.Entry(fen_row, textvariable=self.fen_var).pack(side="left", fill="x", expand=True)
        ttk.Button(fen_row, text="Copiar FEN", command=self.copy_fen).pack(side="left", padx=(6, 0))

        # Engine configuration
        ttk.Label(left_col, text="Engine UCI (vazio = avaliador tatico local com busca de capturas):").pack(
            anchor="w", pady=(6, 2)
        )
        engine_row = ttk.Frame(left_col)
        engine_row.pack(fill="x")
        self.engine_var = tk.StringVar(value=engine_path)
        ttk.Entry(engine_row, textvariable=self.engine_var).pack(side="left", fill="x", expand=True)
        ttk.Button(engine_row, text="Procurar...", command=self.browse_engine).pack(side="left", padx=(6, 0))

        options_row = ttk.Frame(left_col)
        options_row.pack(fill="x", pady=(6, 4))
        ttk.Label(options_row, text="Profundidade:").pack(side="left")
        self.depth_var = tk.IntVar(value=12)
        ttk.Spinbox(options_row, from_=1, to=30, textvariable=self.depth_var, width=5).pack(side="left", padx=(4, 14))

        self.auto_monitor_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row,
            text="Monitorar mudancas na tela (1.5s) automaticamente",
            variable=self.auto_monitor_var,
            command=self.toggle_monitoring,
        ).pack(side="left")

        # Lichess / Chess.com style Engine Lines Panel
        self.lines_frame = ttk.LabelFrame(left_col, text="Melhores Lances da Engine (Multi-PV)", padding=8)
        self.lines_frame.pack(fill="x", pady=6)
        self.line_rows: List[Dict[str, object]] = []
        for i in range(3):
            row_f = ttk.Frame(self.lines_frame)
            row_f.pack(fill="x", pady=3)
            eval_lbl = tk.Label(
                row_f,
                text=" 0.00 ",
                font=("Consolas", 10, "bold"),
                bg="#e8e8e8",
                fg="#1a1a1a",
                width=7,
                relief="flat",
                padx=4,
                pady=2,
            )
            eval_lbl.pack(side="left", padx=(0, 8))

            icon_lbl = ttk.Label(row_f)
            icon_lbl.pack(side="left", padx=(0, 6))

            move_lbl = ttk.Label(row_f, text="-", font=("Segoe UI Symbol", 12, "bold"), width=14)
            move_lbl.pack(side="left", padx=(0, 8))

            pv_lbl = ttk.Label(row_f, text="Clique em Analisar para calcular as melhores linhas", foreground="#555555")
            pv_lbl.pack(side="left", fill="x", expand=True)

            self.line_rows.append({
                "eval": eval_lbl,
                "icon": icon_lbl,
                "move": move_lbl,
                "pv": pv_lbl,
            })

        # Training mode section
        train_frame = ttk.LabelFrame(left_col, text="Modo de Treino", padding=8)
        train_frame.pack(fill="x", pady=4)
        self.training_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            train_frame,
            text="Ocultar melhores lances e setas para treinar",
            variable=self.training_mode_var,
            command=self._refresh_analysis_display,
        ).pack(anchor="w")
        guess_row = ttk.Frame(train_frame)
        guess_row.pack(fill="x", pady=(6, 0))
        ttk.Label(guess_row, text="Seu lance (ex: Cf3, Nf3, Bc4, e2e4):").pack(side="left")
        self.guess_var = tk.StringVar()
        ttk.Entry(guess_row, textvariable=self.guess_var, width=10).pack(side="left", padx=6)
        ttk.Button(guess_row, text="Verificar lance", command=self.check_guess).pack(side="left")
        ttk.Button(guess_row, text="Revelar resposta", command=self.reveal_best_move).pack(side="left", padx=6)
        self.score_label_var = tk.StringVar(value="Acertos: 0 | Erros: 0")
        ttk.Label(guess_row, textvariable=self.score_label_var).pack(side="right")

        action_row = ttk.Frame(left_col)
        action_row.pack(fill="x", pady=8)
        ttk.Button(action_row, text="Analisar posicao", command=self.analyse).pack(side="left")
        ttk.Button(
            action_row,
            text="Capturar + Reconhecer + Analisar (F8)",
            command=self.full_pipeline,
        ).pack(side="left", padx=8)

        self.bind("<F8>", lambda _e: self.full_pipeline())

        self.status = tk.StringVar(
            value="Pronto. Use 'Selecionar area com mouse ✂' ou 'Capturar + Reconhecer + Analisar (F8)'."
        )
        ttk.Label(frame, textvariable=self.status, wraplength=980, font=("Segoe UI Symbol", 10, "bold")).pack(
            anchor="w", pady=(4, 0)
        )

    def _toggle_topmost(self) -> None:
        self.attributes("-topmost", bool(self.topmost_var.get()))

    def copy_fen(self) -> None:
        fen = self.fen_var.get().strip()
        if fen:
            self.clipboard_clear()
            self.clipboard_append(fen)
            self.status.set("FEN copiada para a area de transferencia!")

    def flip_turn(self) -> None:
        """Manually flips side to move ('w' <-> 'b'), updates FEN, and re-runs analysis."""
        import chess

        new_turn = "b" if self.turn_var.get() == "w" else "w"
        self.turn_var.set(new_turn)
        try:
            b = chess.Board(self.fen_var.get().strip())
            b.turn = chess.WHITE if new_turn == "w" else chess.BLACK
            self.fen_var.set(b.fen())
            self._update_turn_banners()
            self.analyse()
        except Exception:
            pass

    def _update_turn_banners(self) -> None:
        my_color = "w" if self.orientation_var.get() == "white" else "b"
        active_turn = self.turn_var.get()
        turn_label = "Brancas (w)" if active_turn == "w" else "Pretas (b)"
        last_str = (
            f" | Ultimo lance na tela: {self.last_detected_move[:2]}→{self.last_detected_move[2:4]}"
            if self.last_detected_move and len(self.last_detected_move) >= 4
            else ""
        )
        if active_turn == my_color:
            self.turn_state_banner.configure(
                text=f"🟢 SUA VEZ DE JOGAR ({turn_label}) — Veja os melhores lances abaixo{last_str}",
                bg="#d4edda",
                fg="#155724",
            )
            self.turn_banner_var.set(f"🟢 SUA VEZ ({turn_label}){last_str}")
            self.lines_frame.configure(text=f"Melhores Lances para VOCE — {turn_label}")
        else:
            self.turn_state_banner.configure(
                text=f"⏳ VOCE JA JOGOU! Aguardando lance do oponente ({turn_label})...{last_str}",
                bg="#fff3cd",
                fg="#856404",
            )
            self.turn_banner_var.set(f"⏳ Voce ja jogou! Vez: {turn_label}{last_str}")
            self.lines_frame.configure(text=f"Respostas Esperadas do Oponente — {turn_label}")

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
            messagebox.showerror("Captura", str(exc))
            return

        def on_selected(selected: Optional[BoardRegion]) -> None:
            self.deiconify()
            if self.topmost_var.get():
                self.attributes("-topmost", True)
            if selected is None:
                self.status.set("Selecao manual cancelada.")
                return
            self.region = selected
            self.region_var.set(f"{selected.x},{selected.y},{selected.width},{selected.height}")
            try:
                self.last_board_image = self.capture.crop_region(self.last_image, selected)
                self._update_preview(self.last_board_image, draw_arrows=False)
                self.status.set(
                    f"Regiao selecionada com o mouse: {self.region_var.get()}. Clique em 'Reconhecer pecas' ou 'F8'."
                )
            except Exception as exc:
                messagebox.showwarning("Regiao", str(exc))

        RegionSelectorOverlay(self, screen_img, on_selected)

    def flip_orientation(self) -> None:
        current = self.orientation_var.get()
        self.orientation_var.set("black" if current == "white" else "white")
        self._update_turn_banners()
        if self.last_board_image is not None:
            self._update_preview(self.last_board_image)

    def _draw_eval_bar(self, score_white_cp: Optional[int], mate_white: Optional[int]) -> None:
        import math

        height = 240
        width = 22
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
        self.eval_bar_canvas.create_rectangle(0, black_h, width, height, fill="#f0f0f0", outline="")
        self.eval_bar_canvas.create_line(0, height // 2, width, height // 2, fill="#c33", width=1)

    def browse_engine(self) -> None:
        path = filedialog.askopenfilename(title="Selecionar executavel da engine UCI")
        if path:
            self.engine_var.set(path)

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

    def _square_rect_px(self, square_name: str, size: int = 240) -> Tuple[int, int, int, int]:
        import chess

        sq = chess.parse_square(square_name)
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        orientation = self.orientation_var.get()
        col = file_idx if orientation == "white" else 7 - file_idx
        row = 7 - rank_idx if orientation == "white" else rank_idx
        sq_size = size / 8.0
        x1 = int(round(col * sq_size))
        y1 = int(round(row * sq_size))
        x2 = int(round((col + 1) * sq_size))
        y2 = int(round((row + 1) * sq_size))
        return x1, y1, x2, y2

    def _square_center_px(self, square_name: str, size: int = 240) -> Tuple[int, int]:
        x1, y1, x2, y2 = self._square_rect_px(square_name, size)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _update_preview(self, board_img, draw_arrows: bool = True) -> None:
        try:
            import cv2
            from PIL import Image, ImageTk

            canvas = cv2.resize(board_img, (240, 240), interpolation=cv2.INTER_AREA)

            if self.last_detected_move and len(self.last_detected_move) >= 4:
                hl = canvas.copy()
                for sq_name in (self.last_detected_move[:2], self.last_detected_move[2:4]):
                    x1, y1, x2, y2 = self._square_rect_px(sq_name, 240)
                    cv2.rectangle(hl, (x1, y1), (x2, y2), (45, 225, 245), -1)
                canvas = cv2.addWeighted(hl, 0.35, canvas, 0.65, 0)

            show_arrows = draw_arrows and bool(self.last_lines) and not self.training_mode_var.get()
            if show_arrows:
                overlay = canvas.copy()
                my_color = "w" if self.orientation_var.get() == "white" else "b"
                is_my_turn = self.turn_var.get() == my_color
                if is_my_turn:
                    arrow_styles = [
                        (0, (40, 175, 75), 6),   # 1st best move (YOUR turn): green BGR
                        (1, (215, 130, 35), 4),  # 2nd best move (YOUR turn): blue/cyan BGR
                    ]
                else:
                    arrow_styles = [
                        (0, (30, 120, 235), 5),  # Opponent's expected reply: orange BGR
                    ]
                for idx, color_bgr, thickness in reversed(arrow_styles):
                    if idx < len(self.last_lines):
                        uci = self.last_lines[idx].uci_move
                        if len(uci) >= 4:
                            p1 = self._square_center_px(uci[:2], 240)
                            p2 = self._square_center_px(uci[2:4], 240)
                            cv2.circle(overlay, p1, 9, color_bgr, -1, lineType=cv2.LINE_AA)
                            cv2.arrowedLine(
                                overlay, p1, p2, color_bgr, thickness, line_type=cv2.LINE_AA, tipLength=0.26
                            )
                canvas = cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0)

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            self.preview_photo = ImageTk.PhotoImage(pil_img)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception:
            pass

    def capture_screen(self) -> None:
        try:
            self.last_image = self.capture.capture()
            self.status.set("Tela capturada. Agora detecte/selecione o tabuleiro ou reconheca as pecas.")
        except Exception as exc:
            messagebox.showerror("Captura", str(exc))

    def detect_board(self) -> None:
        if self.last_image is None:
            self.capture_screen()
        if self.last_image is None:
            return
        try:
            self.region = self.detector.detect(self.last_image)
            self.region_var.set(f"{self.region.x},{self.region.y},{self.region.width},{self.region.height}")
            self.last_board_image = self.capture.crop_region(self.last_image, self.region)
            self._update_preview(self.last_board_image, draw_arrows=False)
            self.status.set(f"Tabuleiro detectado automaticamente em {self.region_var.get()}.")
        except Exception as exc:
            messagebox.showwarning("Deteccao", str(exc))

    def calibrate_from_screen(self) -> None:
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
            self._update_preview(board_img, draw_arrows=False)
            # Always calibrate using DEFAULT_START_FEN so a previous modified FEN in the text box doesn't corrupt calibration
            count = self.recognizer.calibrate(
                board_img, fen=DEFAULT_START_FEN, orientation=self.orientation_var.get()
            )
            self.fen_var.set(DEFAULT_START_FEN)
            self.prev_board_fen = DEFAULT_START_BOARD_FEN
            self.calib_badge.set(f"Calibrado ({count} tipos de peca)")
            self.status.set(
                f"Calibracao concluida com {count} tipos de peca! Os templates foram salvos localmente."
            )
        except Exception as exc:
            messagebox.showerror("Calibracao", str(exc))

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
            self.turn_var.set(inferred_turn)
            if last_mv:
                self.last_detected_move = last_mv
            elif raw_res.board_fen == DEFAULT_START_BOARD_FEN:
                self.last_detected_move = None
            raw_res.inferred_turn = inferred_turn
            raw_res.last_move_uci = self.last_detected_move

            b = chess.Board(raw_res.fen)
            b.turn = chess.WHITE if inferred_turn == "w" else chess.BLACK
            raw_res.fen = b.fen()

        self.prev_board_fen = raw_res.board_fen
        self._update_turn_banners()
        return raw_res

    def recognize_board(self) -> Optional[RecognitionResult]:
        if self.last_image is None:
            self.capture_screen()
        if self.last_image is None:
            return None
        try:
            region = self._get_or_parse_region()
            board_img = self.capture.crop_region(self.last_image, region)
            self.last_board_image = board_img
            self.last_lines = []
            result = self._apply_smart_recognition(board_img)
            self._update_preview(board_img, draw_arrows=False)
            self.fen_var.set(result.fen)
            pct = int(round(result.confidence * 100))
            if result.uncertain_squares:
                sq_list = ", ".join(result.uncertain_squares[:6])
                self.confidence_var.set(f"Confianca: {pct}% (incerteza: {sq_list})")
                self.status.set(
                    f"Posicao reconhecida ({pct}% confianca). Casas incertas: {sq_list}. "
                    + ("Use 'Calibrar posicao inicial' para maior precisao." if not result.calibrated else "")
                )
            else:
                self.confidence_var.set(f"Confianca: {pct}% (alta)")
                self.status.set(f"FEN reconhecida automaticamente ({pct}% confianca). {self.turn_banner_var.get()}")
            return result
        except Exception as exc:
            messagebox.showwarning("Reconhecimento", str(exc))
            return None

    def full_pipeline(self) -> None:
        self.capture_screen()
        if self.last_image is None:
            return
        res = self.recognize_board()
        if res is not None:
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
                self.last_board_image = board_img
                self._update_preview(board_img, draw_arrows=False)
                self.fen_var.set(res.fen)
                self.analyse()
        except Exception as exc:
            self.status.set(f"Erro no monitoramento: {exc}")
        self.monitoring_job = self.after(1500, self._schedule_monitor)

    def check_guess(self) -> None:
        guess_raw = self.guess_var.get().strip()
        if not guess_raw:
            messagebox.showinfo("Treino", "Digite seu lance (ex: Cf3, Nf3, Bc4 ou e2e4).")
            return
        if not self.last_best_move:
            messagebox.showinfo("Treino", "Clique em 'Analisar posicao' primeiro para calcular o melhor lance.")
            return
        try:
            guess_uci = parse_user_move_to_uci(self.fen_var.get().strip(), guess_raw)
        except ValueError as exc:
            messagebox.showwarning("Treino", str(exc))
            return

        if guess_uci == self.last_best_move.lower():
            self.training_hits += 1
            self.reveal_best_move()
            self.status.set(f"Correto! {self.last_analysis_text}")
        else:
            self.training_misses += 1
            top = self.last_lines[0] if self.last_lines else None
            piece_name = PT_PIECE_NAMES.get(top.piece_symbol.upper(), "peca") if top else "peca"
            hint_sq = self.last_best_move[:2]
            self.status.set(f"Lance diferente do sugerido. Dica: jogue com {piece_name} da casa {hint_sq}.")
        self.score_label_var.set(f"Acertos: {self.training_hits} | Erros: {self.training_misses}")

    def reveal_best_move(self) -> None:
        if not self.last_lines:
            self.status.set("Nenhuma analise disponivel ainda.")
            return
        self._render_engine_lines(self.last_lines, force_show=True)
        if self.last_board_image is not None:
            self._update_preview(self.last_board_image, draw_arrows=True)
        self.status.set(self.last_analysis_text)

    def _render_engine_lines(self, lines: List[EngineLine], force_show: bool = False) -> None:
        hide = self.training_mode_var.get() and not force_show
        if lines and not hide:
            self._draw_eval_bar(lines[0].score_white_cp, lines[0].mate_white)
        else:
            self._draw_eval_bar(0, None)

        for idx, row_widgets in enumerate(self.line_rows):
            eval_lbl: tk.Label = row_widgets["eval"]  # type: ignore[assignment]
            icon_lbl: ttk.Label = row_widgets["icon"]  # type: ignore[assignment]
            move_lbl: ttk.Label = row_widgets["move"]  # type: ignore[assignment]
            pv_lbl: ttk.Label = row_widgets["pv"]  # type: ignore[assignment]

            if hide:
                eval_lbl.configure(text=" ? ", bg="#d8d8d8", fg="#333333")
                icon_lbl.configure(image="")
                move_lbl.configure(text="Lance oculto")
                pv_lbl.configure(text="Modo treino ativo — digite seu palpite ou clique em 'Revelar resposta'")
                continue

            if idx < len(lines):
                line = lines[idx]
                white_ahead = (
                    (line.mate_white is not None and line.mate_white > 0)
                    or (line.score_white_cp is not None and line.score_white_cp >= 0)
                )
                bg_col = "#f0f0f0" if white_ahead else "#262421"
                fg_col = "#1a1a1a" if white_ahead else "#ffffff"
                eval_lbl.configure(text=f" {line.eval_text} ", bg=bg_col, fg=fg_col)

                icon = self.piece_icons.get(line.piece_symbol)
                if icon is not None:
                    icon_lbl.configure(image=icon)
                else:
                    icon_lbl.configure(image="")

                origin, dest = line.uci_move[:2], line.uci_move[2:4]
                move_lbl.configure(text=f"{line.figurine_move}  ({origin}→{dest})")
                pv_lbl.configure(text=line.pv_figurine)
            else:
                eval_lbl.configure(text=" - ", bg="#e8e8e8", fg="#777777")
                icon_lbl.configure(image="")
                move_lbl.configure(text="-")
                pv_lbl.configure(text="")

    def _refresh_analysis_display(self) -> None:
        self._render_engine_lines(self.last_lines)
        if self.last_board_image is not None:
            self._update_preview(self.last_board_image, draw_arrows=True)

    def _on_analysis_success(self, lines: List[EngineLine]) -> None:
        self.last_lines = lines
        top = lines[0]
        self.last_best_move = top.uci_move
        piece_name = PT_PIECE_NAMES.get(top.piece_symbol.upper(), "Peca")
        self.last_analysis_text = (
            f"Melhor lance: {top.figurine_move} ({piece_name} {top.uci_move[:2]}→{top.uci_move[2:4]}) "
            f"| Avaliacao: {top.eval_text} | Linha: {top.pv_figurine}"
        )
        self._render_engine_lines(lines)
        if self.last_board_image is not None:
            self._update_preview(self.last_board_image, draw_arrows=True)

        if self.training_mode_var.get():
            msg = "Posicao analisada! Modo treino ativo: digite seu lance (ex: Cf3, Nf3, e2e4) e clique em 'Verificar lance'."
        else:
            msg = self.last_analysis_text
        self.status.set(msg)

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                req_id, kind, payload = self._ui_queue.get_nowait()
                if req_id != self._analysis_seq:
                    # Descarta resultado obsoleto de uma analise anterior
                    continue
                if kind == "ok":
                    self._on_analysis_success(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    msg = str(payload)
                    self.status.set(f"Erro na analise: {msg}")
                    messagebox.showerror("Analise", msg)
        except queue.Empty:
            pass
        try:
            if self.winfo_exists():
                self.after(50, self._poll_ui_queue)
        except tk.TclError:
            pass

    def analyse(self) -> None:
        try:
            fen = self.fen_var.get().strip()
            depth = int(self.depth_var.get())
            engine_path = self.engine_var.get().strip()
        except Exception as exc:
            messagebox.showerror("Analise", f"Parametros invalidos: {exc}")
            return

        self._analysis_seq += 1
        req_id = self._analysis_seq
        ui_queue = self._ui_queue
        self.status.set("Analisando posicao...")

        def work() -> None:
            try:
                if engine_path:
                    lines = UciEngine(engine_path).analyse_lines(fen, depth, multipv=3)
                else:
                    lines = LocalChessEvaluator.analyse_lines(fen, depth, multipv=3)
                ui_queue.put((req_id, "ok", lines))
            except Exception as exc:
                ui_queue.put((req_id, "error", str(exc)))

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess Coach: captura de tela, visao computacional e analise UCI")
    parser.add_argument(
        "--engine",
        default=find_default_engine(),
        help="caminho para uma engine UCI, como Stockfish ou OpenTal (auto-detectado por padrao)",
    )
    args = parser.parse_args()
    ChessCoachApp(args.engine).mainloop()


if __name__ == "__main__":
    main()
