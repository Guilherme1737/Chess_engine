"""Local screen-capture chess coach with piece recognition and UCI analysis."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration_data.npz"
PIECE_TYPES = ("P", "N", "B", "R", "Q", "K")


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
            if len(polygon) == 4 and 0.75 <= ratio <= 1.33 and area > image_width * image_height * 0.04:
                candidates.append((area, BoardRegion(x, y, width, height)))
        if not candidates:
            raise RuntimeError("Nenhum tabuleiro foi detectado; informe x,y,w,h manualmente")
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
        # Maps symbol ('P','N','B','R','Q','K','p','n','b','r','q','k') -> dict of feature arrays
        self.templates: Dict[str, Dict[str, object]] = {}
        self.calibrated = False
        self._synthetic_masks = self._build_synthetic_masks()
        if self.calibration_path and self.calibration_path.is_file():
            try:
                self.load_calibration(self.calibration_path)
            except Exception:
                pass

    @staticmethod
    def _build_synthetic_masks() -> Dict[str, object]:
        """Builds 52x52 synthetic silhouette templates for fallback recognition."""
        import cv2
        import numpy as np

        size = 52
        masks: Dict[str, object] = {}

        # Pawn: small round head, narrow neck, medium base
        p = np.zeros((size, size), dtype=np.float32)
        cv2.circle(p, (26, 17), 7, 1.0, -1)
        cv2.fillPoly(p, [np.array([[20, 24], [32, 24], [35, 39], [17, 39]])], 1.0)
        cv2.rectangle(p, (13, 39), (39, 45), 1.0, -1)
        masks["P"] = p

        # Rook: flat crenellated top, wide straight tower, wide base
        r = np.zeros((size, size), dtype=np.float32)
        cv2.rectangle(r, (14, 11), (38, 18), 1.0, -1)
        cv2.rectangle(r, (18, 18), (34, 38), 1.0, -1)
        cv2.rectangle(r, (12, 38), (40, 45), 1.0, -1)
        masks["R"] = r

        # Knight: asymmetric curved profile
        n = np.zeros((size, size), dtype=np.float32)
        pts_n = np.array([[16, 44], [38, 44], [36, 26], [33, 12], [22, 10], [13, 21], [21, 25], [16, 36]])
        cv2.fillPoly(n, [pts_n], 1.0)
        masks["N"] = n

        # Bishop: pointed dome top, slender body, base
        b = np.zeros((size, size), dtype=np.float32)
        cv2.circle(b, (26, 10), 3, 1.0, -1)
        cv2.ellipse(b, (26, 24), (9, 12), 0, 0, 360, 1.0, -1)
        cv2.rectangle(b, (13, 39), (39, 45), 1.0, -1)
        masks["B"] = b

        # Queen: wide crown with multiple spikes
        q = np.zeros((size, size), dtype=np.float32)
        pts_q = np.array([
            [14, 40], [10, 16], [19, 26], [26, 12], [33, 26], [42, 16], [38, 40]
        ])
        cv2.fillPoly(q, [pts_q], 1.0)
        cv2.rectangle(q, (12, 39), (40, 45), 1.0, -1)
        masks["Q"] = q

        # King: tall crown with top cross
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
                # Use outer 4px ring of the inner patch to sample square background color
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

    def _extract_features(self, patch, bg_color) -> Dict[str, object]:
        import cv2
        import numpy as np

        patch_f = patch.astype(np.float32)
        # Per-square local background refinement using outer 3px corners
        corners = np.vstack([
            patch_f[:4, :4].reshape(-1, 3),
            patch_f[:4, -4:].reshape(-1, 3),
            patch_f[-4:, :4].reshape(-1, 3),
            patch_f[-4:, -4:].reshape(-1, 3),
        ])
        local_bg = 0.5 * bg_color + 0.5 * np.median(corners, axis=0)

        diff = np.linalg.norm(patch_f - local_bg[None, None, :], axis=2)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)

        # Foreground mask: pixels different from square background, with interior holes filled
        fg_raw = (diff > 20.0).astype(np.uint8)
        contours, _ = cv2.findContours(fg_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fg_filled = np.zeros_like(fg_raw)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 6:
                cv2.drawContours(fg_filled, [cnt], -1, 1, thickness=cv2.FILLED)
        fg_mask = fg_filled.astype(np.float32)

        # Eroded interior for reliable piece body color (excludes outline/anti-aliasing border)
        interior = cv2.erode(fg_filled, np.ones((3, 3), np.uint8), iterations=1)
        if np.any(interior > 0):
            body_luma = float(np.median(gray[interior > 0]))
        elif np.any(fg_filled > 0):
            body_luma = float(np.median(gray[fg_filled > 0]))
        else:
            body_luma = 128.0

        # Neutralized grayscale: replace background with 128 so light/dark square background vanishes
        neutral_gray = np.where(fg_mask > 0.5, gray, 128.0).astype(np.float32) / 255.0
        norm_edges = edges / (np.max(edges) + 1e-5)

        return {
            "diff_mean": float(np.mean(diff)),
            "edge_mean": float(np.mean(edges)),
            "fg_ratio": float(np.mean(fg_mask)),
            "fg_mask": fg_mask,
            "neutral_gray": neutral_gray,
            "edges": norm_edges,
            "gray": gray,
            "body_luma": body_luma,
        }

    def calibrate(self, board_image, fen: str = DEFAULT_START_FEN, orientation: str = "white") -> int:
        """Extracts and saves piece templates from a board image with a known FEN."""
        import chess
        import numpy as np

        board = chess.Board(fen)
        norm_board = self._normalize_board(board_image)

        empty_light = []
        empty_dark = []
        for r in range(8):
            for c in range(8):
                file_idx = c if orientation == "white" else 7 - c
                rank_idx = 7 - r if orientation == "white" else r
                sq = chess.square(file_idx, rank_idx)
                piece = board.piece_at(sq)
                patch = self._extract_square_patch(norm_board, r, c).astype(np.float32)
                if piece is None:
                    if (r + c) % 2 == 0:
                        empty_light.append(np.median(patch.reshape(-1, 3), axis=0))
                    else:
                        empty_dark.append(np.median(patch.reshape(-1, 3), axis=0))

        est_light, est_dark = self._estimate_backgrounds(norm_board)
        self.bg_light = np.median(empty_light, axis=0).astype(np.float32) if empty_light else est_light
        self.bg_dark = np.median(empty_dark, axis=0).astype(np.float32) if empty_dark else est_dark

        grouped: Dict[str, List[Dict[str, object]]] = {}
        for r in range(8):
            for c in range(8):
                file_idx = c if orientation == "white" else 7 - c
                rank_idx = 7 - r if orientation == "white" else r
                sq = chess.square(file_idx, rank_idx)
                piece = board.piece_at(sq)
                if piece is None:
                    continue
                symbol = piece.symbol()
                bg = self.bg_light if (r + c) % 2 == 0 else self.bg_dark
                patch = self._extract_square_patch(norm_board, r, c)
                feats = self._extract_features(patch, bg)
                grouped.setdefault(symbol, []).append(feats)

        if not grouped:
            raise ValueError("Nenhuma peca encontrada na FEN informada para calibracao")

        self.templates = {}
        for symbol, items in grouped.items():
            self.templates[symbol] = {
                "fg_mask": np.mean([it["fg_mask"] for it in items], axis=0).astype(np.float32),
                "neutral_gray": np.mean([it["neutral_gray"] for it in items], axis=0).astype(np.float32),
                "edges": np.mean([it["edges"] for it in items], axis=0).astype(np.float32),
            }

        self.calibrated = True
        if self.calibration_path:
            self.save_calibration(self.calibration_path)
        return len(self.templates)

    def save_calibration(self, path: Path) -> None:
        import numpy as np

        if not self.calibrated or self.bg_light is None or self.bg_dark is None:
            return
        arrays: Dict[str, object] = {
            "bg_light": self.bg_light,
            "bg_dark": self.bg_dark,
            "symbols": np.array(list(self.templates.keys())),
        }
        for sym, data in self.templates.items():
            arrays[f"{sym}_fg_mask"] = data["fg_mask"]
            arrays[f"{sym}_neutral_gray"] = data["neutral_gray"]
            arrays[f"{sym}_edges"] = data["edges"]
        np.savez_compressed(path, **arrays)

    def load_calibration(self, path: Path) -> None:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            self.bg_light = data["bg_light"].astype(np.float32)
            self.bg_dark = data["bg_dark"].astype(np.float32)
            symbols = [str(s) for s in data["symbols"]]
            self.templates = {}
            for sym in symbols:
                self.templates[sym] = {
                    "fg_mask": data[f"{sym}_fg_mask"].astype(np.float32),
                    "neutral_gray": data[f"{sym}_neutral_gray"].astype(np.float32),
                    "edges": data[f"{sym}_edges"].astype(np.float32),
                }
            self.calibrated = bool(self.templates)

    @staticmethod
    def _similarity(a, b) -> float:
        import cv2
        import numpy as np

        # Allow small +/-2 pixel shift alignment by matching center crop of b inside a
        margin = 2
        tpl = b[margin:-margin, margin:-margin].astype(np.float32)
        src = a.astype(np.float32)
        if np.std(src) < 1e-5 or np.std(tpl) < 1e-5:
            return 0.0
        res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
        return float(np.max(res))

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
        bg_light = self.bg_light if self.bg_light is not None else est_light
        bg_dark = self.bg_dark if self.bg_dark is not None else est_dark

        square_feats: Dict[Tuple[int, int], Dict[str, object]] = {}
        occupied_coords: List[Tuple[int, int]] = []
        color_scores: Dict[Tuple[int, int], float] = {}

        for r in range(8):
            for c in range(8):
                bg = bg_light if (r + c) % 2 == 0 else bg_dark
                patch = self._extract_square_patch(norm_board, r, c)
                feats = self._extract_features(patch, bg)
                square_feats[(r, c)] = feats

                is_occupied = feats["fg_ratio"] > 0.08 and (feats["diff_mean"] > 10.0 or feats["edge_mean"] > 18.0)
                if is_occupied:
                    occupied_coords.append((r, c))
                    color_scores[(r, c)] = feats["body_luma"]

        # Adaptive threshold for white vs black pieces when uncalibrated
        if color_scores:
            vals = list(color_scores.values())
            color_threshold = 0.5 * (min(vals) + max(vals)) if (max(vals) - min(vals)) > 25.0 else 128.0
        else:
            color_threshold = 128.0

        grid: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]
        confidences: List[float] = []
        uncertain_squares: List[str] = []

        for r in range(8):
            for c in range(8):
                feats = square_feats[(r, c)]
                file_idx = c if orientation == "white" else 7 - c
                rank_idx = 7 - r if orientation == "white" else r
                sq_name = chess.square_name(chess.square(file_idx, rank_idx))

                if (r, c) not in occupied_coords:
                    confidences.append(0.98)
                    continue

                if self.calibrated and self.templates:
                    best_sym = None
                    best_score = -1.0
                    second_score = -1.0
                    for sym, tpl in self.templates.items():
                        s_gray = self._similarity(feats["neutral_gray"], tpl["neutral_gray"])
                        s_mask = self._similarity(feats["fg_mask"], tpl["fg_mask"])
                        s_edge = self._similarity(feats["edges"], tpl["edges"])
                        score = 0.50 * s_gray + 0.30 * s_mask + 0.20 * s_edge
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
                    is_white = color_scores[(r, c)] >= color_threshold
                    best_type = "P"
                    best_score = -1.0
                    for p_type, syn_mask in self._synthetic_masks.items():
                        score = self._similarity(feats["fg_mask"], syn_mask)
                        if score > best_score:
                            best_score = score
                            best_type = p_type
                    sym = best_type if is_white else best_type.lower()
                    conf = float(np.clip(best_score * 0.75, 0.25, 0.75))
                    grid[r][c] = sym
                    confidences.append(conf)
                    if conf < 0.55:
                        uncertain_squares.append(sq_name)

        # Build board according to orientation
        board = chess.Board.empty()
        for r in range(8):
            for c in range(8):
                sym = grid[r][c]
                if not sym:
                    continue
                file_idx = c if orientation == "white" else 7 - c
                rank_idx = 7 - r if orientation == "white" else r
                sq = chess.square(file_idx, rank_idx)
                board.set_piece_at(sq, chess.Piece.from_symbol(sym))

        board.turn = chess.WHITE if side_to_move == "w" else chess.BLACK
        board.set_castling_fen(castling if castling else "-")
        board_fen = board.board_fen()
        full_fen = board.fen()
        overall_conf = float(np.mean(confidences)) if confidences else 0.0

        return RecognitionResult(
            fen=full_fen,
            board_fen=board_fen,
            confidence=overall_conf,
            uncertain_squares=uncertain_squares,
            calibrated=self.calibrated,
        )


class LocalChessEvaluator:
    """Lightweight alpha-beta evaluator using python-chess as a fallback when no UCI binary is configured."""

    PIECE_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}

    @classmethod
    def evaluate_board(cls, board) -> int:
        import chess

        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        score = 0
        for sq, piece in board.piece_map().items():
            val = cls.PIECE_VALUES.get(piece.piece_type, 0)
            # Center control bonus
            file_idx = chess.square_file(sq)
            rank_idx = chess.square_rank(sq)
            center_bonus = int(12 - 2 * (abs(3.5 - file_idx) + abs(3.5 - rank_idx)))
            total = val + center_bonus
            score += total if piece.color == chess.WHITE else -total
        return score if board.turn == chess.WHITE else -score

    @classmethod
    def analyse(cls, fen: str, depth: int = 3) -> Tuple[str, str]:
        import chess

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise RuntimeError("A posicao informada nao possui lances legais")

        search_depth = max(1, min(depth, 3))

        def negamax(b, d: int, alpha: int, beta: int) -> int:
            if d == 0 or b.is_game_over():
                return cls.evaluate_board(b)
            max_eval = -1000000
            for move in b.legal_moves:
                b.push(move)
                val = -negamax(b, d - 1, -beta, -alpha)
                b.pop()
                if val > max_eval:
                    max_eval = val
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    break
            return max_eval

        best_move = legal_moves[0]
        best_score = -1000000
        alpha = -1000000
        beta = 1000000
        for move in legal_moves:
            board.push(move)
            score = -negamax(board, search_depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        uci_move = best_move.uci()
        return uci_move, f"Melhor lance: {uci_move} | avaliacao: {best_score} cp (avaliador local)"


class UciEngine:
    def __init__(self, executable: str):
        self.executable = executable

    def analyse_with_move(self, fen: str, depth: int) -> Tuple[str, str]:
        if not self.executable:
            raise RuntimeError("Informe o caminho de uma engine UCI, por exemplo OpenTal")
        executable = self.executable if os.path.isfile(self.executable) else shutil.which(self.executable)
        if not executable or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Engine nao executavel: {self.executable}")
        commands = [
            "uci",
            "isready",
            f"position fen {fen}",
            f"go depth {depth}",
        ]
        process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        for command in commands:
            process.stdin.write(command + "\n")
        process.stdin.flush()
        best_move = None
        score = None
        for line in process.stdout:
            if line.startswith("info"):
                if "score cp" in line:
                    score = line.split("score cp", 1)[1].split()[0] + " cp"
                elif "score mate" in line:
                    mate_val = line.split("score mate", 1)[1].split()[0]
                    score = f"mate em {mate_val}"
            if line.startswith("bestmove"):
                best_move = line.split()[1]
                break
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=5)
        if not best_move:
            raise RuntimeError("A engine nao retornou bestmove")
        summary = f"Melhor lance: {best_move} | avaliacao: {score or 'indisponivel'}"
        return best_move, summary

    def analyse(self, fen: str, depth: int) -> str:
        _, summary = self.analyse_with_move(fen, depth)
        return summary


class ChessCoachApp(tk.Tk):
    def __init__(self, engine_path: str):
        super().__init__()
        self.title("Chess Coach")
        self.geometry("880x680")
        self.capture = ScreenCapture()
        self.detector = BoardDetector()
        self.recognizer = PieceRecognizer()
        self.last_image = None
        self.last_board_image = None
        self.region: Optional[BoardRegion] = None
        self.preview_photo = None
        self.last_best_move: Optional[str] = None
        self.last_analysis_text: str = ""
        self.training_hits = 0
        self.training_misses = 0
        self.monitoring_job: Optional[str] = None
        self._build_ui(engine_path)

    def _build_ui(self, engine_path: str) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)

        header = ttk.Frame(frame)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Chess Coach", font=("TkDefaultFont", 20, "bold")).pack(side="left")
        self.calib_badge = tk.StringVar(
            value="Calibrado" if self.recognizer.calibrated else "Nao calibrado (usando silhuetas padrao)"
        )
        ttk.Label(header, textvariable=self.calib_badge).pack(side="right")
        ttk.Label(
            frame,
            text="Captura local + deteccao de tabuleiro + reconhecimento automatico de pecas (OpenCV) + engine UCI",
        ).pack(anchor="w", pady=(0, 12))

        main_pane = ttk.Frame(frame)
        main_pane.pack(fill="both", expand=True)

        left_col = ttk.Frame(main_pane)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 14))

        right_col = ttk.LabelFrame(main_pane, text="Preview do Tabuleiro", padding=8)
        right_col.pack(side="right", fill="y")
        self.preview_label = ttk.Label(right_col, text="Nenhum tabuleiro\ncapturado ainda", anchor="center", width=28)
        self.preview_label.pack(padx=6, pady=6)
        self.confidence_var = tk.StringVar(value="Confianca: -")
        ttk.Label(right_col, textvariable=self.confidence_var).pack(anchor="w", pady=(6, 2))

        # Capture & Board Region controls
        controls = ttk.Frame(left_col)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Button(controls, text="Capturar tela", command=self.capture_screen).pack(side="left")
        ttk.Button(controls, text="Detectar tabuleiro", command=self.detect_board).pack(side="left", padx=6)
        ttk.Button(controls, text="Reconhecer pecas", command=self.recognize_board).pack(side="left", padx=6)
        ttk.Button(controls, text="Calibrar da posicao inicial", command=self.calibrate_from_screen).pack(side="left")

        region_row = ttk.Frame(left_col)
        region_row.pack(fill="x", pady=6)
        ttk.Label(region_row, text="Regiao x,y,w,h:").pack(side="left")
        self.region_var = tk.StringVar()
        ttk.Entry(region_row, textvariable=self.region_var, width=18).pack(side="left", padx=6)
        ttk.Label(region_row, text="Orientacao:").pack(side="left", padx=(8, 4))
        self.orientation_var = tk.StringVar(value="white")
        ttk.Combobox(
            region_row,
            textvariable=self.orientation_var,
            values=("white", "black"),
            state="readonly",
            width=8,
        ).pack(side="left")
        ttk.Label(region_row, text="Vez:").pack(side="left", padx=(8, 4))
        self.turn_var = tk.StringVar(value="w")
        ttk.Combobox(
            region_row,
            textvariable=self.turn_var,
            values=("w", "b"),
            state="readonly",
            width=4,
        ).pack(side="left")

        # FEN input/output
        ttk.Label(left_col, text="FEN da posicao (gerada automaticamente ou editada manualmente):").pack(
            anchor="w", pady=(10, 4)
        )
        self.fen_var = tk.StringVar(value=DEFAULT_START_FEN)
        ttk.Entry(left_col, textvariable=self.fen_var).pack(fill="x")

        # Engine configuration
        ttk.Label(left_col, text="Engine UCI (deixe vazio para usar o avaliador local embutido):").pack(
            anchor="w", pady=(10, 4)
        )
        engine_row = ttk.Frame(left_col)
        engine_row.pack(fill="x")
        self.engine_var = tk.StringVar(value=engine_path)
        ttk.Entry(engine_row, textvariable=self.engine_var).pack(side="left", fill="x", expand=True)
        ttk.Button(engine_row, text="Procurar...", command=self.browse_engine).pack(side="left", padx=(6, 0))

        options_row = ttk.Frame(left_col)
        options_row.pack(fill="x", pady=(10, 4))
        ttk.Label(options_row, text="Profundidade:").pack(side="left")
        self.depth_var = tk.IntVar(value=12)
        ttk.Spinbox(options_row, from_=1, to=30, textvariable=self.depth_var, width=5).pack(side="left", padx=(4, 14))

        self.auto_monitor_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_row,
            text="Monitorar mudancas na tela (2s)",
            variable=self.auto_monitor_var,
            command=self.toggle_monitoring,
        ).pack(side="left")

        # Training mode section
        train_frame = ttk.LabelFrame(left_col, text="Modo de Treino", padding=8)
        train_frame.pack(fill="x", pady=10)
        self.training_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            train_frame,
            text="Ocultar melhor lance e treinar",
            variable=self.training_mode_var,
        ).pack(anchor="w")
        guess_row = ttk.Frame(train_frame)
        guess_row.pack(fill="x", pady=(6, 0))
        ttk.Label(guess_row, text="Seu lance (UCI, ex: e2e4):").pack(side="left")
        self.guess_var = tk.StringVar()
        ttk.Entry(guess_row, textvariable=self.guess_var, width=10).pack(side="left", padx=6)
        ttk.Button(guess_row, text="Verificar lance", command=self.check_guess).pack(side="left")
        ttk.Button(guess_row, text="Revelar resposta", command=self.reveal_best_move).pack(side="left", padx=6)
        self.score_label_var = tk.StringVar(value="Acertos: 0 | Erros: 0")
        ttk.Label(guess_row, textvariable=self.score_label_var).pack(side="right")

        action_row = ttk.Frame(left_col)
        action_row.pack(fill="x", pady=10)
        ttk.Button(action_row, text="Analisar posicao", command=self.analyse).pack(side="left")
        ttk.Button(
            action_row,
            text="Capturar + Reconhecer + Analisar",
            command=self.full_pipeline,
        ).pack(side="left", padx=8)

        self.status = tk.StringVar(value="Pronto. Capture a tela ou use 'Capturar + Reconhecer + Analisar'.")
        ttk.Label(frame, textvariable=self.status, wraplength=820, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(6, 0)
        )

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

    def _update_preview(self, board_img) -> None:
        try:
            import cv2
            from PIL import Image, ImageTk

            rgb = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((220, 220))
            self.preview_photo = ImageTk.PhotoImage(pil_img)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception:
            pass

    def capture_screen(self) -> None:
        try:
            self.last_image = self.capture.capture()
            self.status.set("Tela capturada. Agora detecte o tabuleiro, calibre ou reconheca as pecas.")
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
            self._update_preview(self.last_board_image)
            self.status.set(f"Tabuleiro detectado em {self.region_var.get()}.")
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
            self._update_preview(board_img)
            calib_fen = self.fen_var.get().strip() or DEFAULT_START_FEN
            count = self.recognizer.calibrate(board_img, fen=calib_fen, orientation=self.orientation_var.get())
            self.calib_badge.set(f"Calibrado ({count} tipos de peca)")
            self.status.set(
                f"Calibracao concluida com {count} tipos de peca! Os templates foram salvos localmente."
            )
        except Exception as exc:
            messagebox.showerror("Calibracao", str(exc))

    def recognize_board(self) -> Optional[RecognitionResult]:
        if self.last_image is None:
            self.capture_screen()
        if self.last_image is None:
            return None
        try:
            region = self._get_or_parse_region()
            board_img = self.capture.crop_region(self.last_image, region)
            self.last_board_image = board_img
            self._update_preview(board_img)
            result = self.recognizer.recognize(
                board_img,
                orientation=self.orientation_var.get(),
                side_to_move=self.turn_var.get(),
            )
            self.fen_var.set(result.fen)
            pct = int(round(result.confidence * 100))
            if result.uncertain_squares:
                sq_list = ", ".join(result.uncertain_squares[:6])
                self.confidence_var.set(f"Confianca: {pct}% (incerteza: {sq_list})")
                self.status.set(
                    f"Posicao reconhecida ({pct}% confianca). Casas incertas: {sq_list}. "
                    + ("Use 'Calibrar da posicao inicial' para maior precisao." if not result.calibrated else "")
                )
            else:
                self.confidence_var.set(f"Confianca: {pct}% (alta)")
                self.status.set(f"FEN reconhecida automaticamente com {pct}% de confianca.")
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
            res = self.recognizer.recognize(
                board_img,
                orientation=self.orientation_var.get(),
                side_to_move=self.turn_var.get(),
            )
            old_board_fen = self.fen_var.get().strip().split()[0] if self.fen_var.get().strip() else ""
            if res.board_fen != old_board_fen:
                self.last_board_image = board_img
                self._update_preview(board_img)
                self.fen_var.set(res.fen)
                self.analyse()
        except Exception:
            pass
        self.monitoring_job = self.after(2000, self._schedule_monitor)

    def check_guess(self) -> None:
        guess = self.guess_var.get().strip().lower()
        if not guess:
            messagebox.showinfo("Treino", "Digite seu lance no formato UCI (ex: e2e4).")
            return
        if not self.last_best_move:
            messagebox.showinfo("Treino", "Clique em 'Analisar posicao' primeiro para calcular o melhor lance.")
            return
        if guess == self.last_best_move.lower():
            self.training_hits += 1
            self.status.set(f"Correto! {self.last_analysis_text}")
        else:
            self.training_misses += 1
            hint_sq = self.last_best_move[:2]
            self.status.set(f"Lance diferente do sugerido. Dica: mova a peca da casa {hint_sq}.")
        self.score_label_var.set(f"Acertos: {self.training_hits} | Erros: {self.training_misses}")

    def reveal_best_move(self) -> None:
        if self.last_analysis_text:
            self.status.set(self.last_analysis_text)
        else:
            self.status.set("Nenhuma analise disponivel ainda.")

    def analyse(self) -> None:
        def work():
            try:
                fen = self.fen_var.get().strip()
                depth = self.depth_var.get()
                engine_path = self.engine_var.get().strip()
                if engine_path:
                    best_move, summary = UciEngine(engine_path).analyse_with_move(fen, depth)
                else:
                    best_move, summary = LocalChessEvaluator.analyse(fen, depth)
                self.last_best_move = best_move
                self.last_analysis_text = summary
                if self.training_mode_var.get():
                    msg = "Posicao analisada! Modo treino ativo: digite seu lance e clique em 'Verificar lance'."
                else:
                    msg = summary
                self.after(0, lambda: self.status.set(msg))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Analise", str(exc)))

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess Coach: captura de tela, visao computacional e analise UCI")
    parser.add_argument("--engine", default="", help="caminho para uma engine UCI, como OpenTal ou Stockfish")
    args = parser.parse_args()
    ChessCoachApp(args.engine).mainloop()


if __name__ == "__main__":
    main()
