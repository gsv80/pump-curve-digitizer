# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "opencv-python",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "pdf2image",
#   "scipy",
# ]
# ///

# ============================================================================
# PUMP CURVE DIGITIZER - Professional Version (Updated)
# ============================================================================

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json
from datetime import datetime
from scipy import interpolate
import os
import re

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CalibrationPoint:
    """A point used for axis calibration."""
    x_px: int
    y_px: int
    value: float
    description: str = ""


@dataclass
class AxisCalibration:
    """Calibration data for all axes using two points per axis."""
    # X-axis (flow rate) - two points
    x_points: Tuple[CalibrationPoint, CalibrationPoint]

    # Y-axis for pump head (left) - two points
    y_head_points: Tuple[CalibrationPoint, CalibrationPoint]

    # Y-axis for NPSH (right) - two points (optional)
    y_npsh_points: Optional[Tuple[CalibrationPoint, CalibrationPoint]] = None

    # Y-axis for efficiency (usually right or secondary) - two points (optional)
    y_efficiency_points: Optional[Tuple[CalibrationPoint, CalibrationPoint]] = None

    def get_x_transform(self):
        """Return linear transformation: real_x = a * x_px + b."""
        (p1, p2) = self.x_points
        a = (p2.value - p1.value) / (p2.x_px - p1.x_px)
        b = p1.value - a * p1.x_px
        return a, b

    def get_y_head_transform(self):
        """Return linear transformation for head axis."""
        (p1, p2) = self.y_head_points
        a = (p2.value - p1.value) / (p2.y_px - p1.y_px)
        b = p1.value - a * p1.y_px
        return a, b

    def get_y_npsh_transform(self):
        """Return linear transformation for NPSH axis, if available."""
        if self.y_npsh_points:
            (p1, p2) = self.y_npsh_points
            a = (p2.value - p1.value) / (p2.y_px - p1.y_px)
            b = p1.value - a * p1.y_px
            return a, b
        return None

    def get_y_efficiency_transform(self):
        """Return linear transformation for efficiency axis, if available."""
        if self.y_efficiency_points:
            (p1, p2) = self.y_efficiency_points
            a = (p2.value - p1.value) / (p2.y_px - p1.y_px)
            b = p1.value - a * p1.y_px
            return a, b
        return None


@dataclass
class Point:
    """A point on a curve (pixel coordinates)."""
    x_px: int
    y_px: int
    # Optional efficiency value if this point lies on an efficiency line
    efficiency: Optional[float] = None


@dataclass
class Curve:
    """Base class for curves."""
    name: str
    points: List[Point] = field(default_factory=list)
    color: Tuple[int, int, int] = (0, 255, 0)

    def add_point(self, x_px: int, y_px: int, efficiency: Optional[float] = None):
        self.points.append(Point(x_px, y_px, efficiency))


@dataclass
class PumpCurve(Curve):
    pump_number: int = 0
    diameter: float = 0.0
    has_npsh: bool = False
    has_efficiency: bool = False

@dataclass
class NPSHCurve(Curve):
    pump_number: int = 0
    diameter: float = 0.0

@dataclass
class EfficiencyPoint:
    pump_curve_name: str
    pump_number: int
    diameter: float
    x_px: int
    y_px: int
    efficiency_value: float


@dataclass
class FittedCurve:
    """Result of curve fitting."""
    name: str
    curve_type: str  # 'pump' or 'npsh'
    points_px: List[Tuple[float, float]]
    points_real: List[Tuple[float, float]]
    coefficients: List[float]  # Polynomial coefficients [a, b, c, d] for ax³+bx²+cx+d
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    rmse: float
    confidence: float
    efficiency_points: List[Tuple[float, float]] = field(default_factory=list)
    # New fields for efficiency curve fitting (4th degree polynomial)
    efficiency_coefficients: Optional[List[float]] = None  # [a4, a3, a2, a1, a0] for a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
    efficiency_rmse: Optional[float] = None
    # NEW: efficiency x-range (min and max from actual efficiency points)
    efficiency_x_min: Optional[float] = None
    efficiency_x_max: Optional[float] = None

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class PumpCurveDigitizer:
    """
    Professional tool for digitizing pump curves from images.
    Manual point selection with automatic curve fitting.
    """

    def __init__(self):
        self.image = None
        self.original_image = None
        self.graph_rect = None
        self.calibration = None

        # Curves data
        self.pump_curves: List[PumpCurve] = []
        self.npsh_curves: List[NPSHCurve] = []
        self.efficiency_points: List[EfficiencyPoint] = []

        # Current state
        self.current_mode = "idle"
        self.current_pump = None
        self.current_curve = None
        self.selected_points = []

        # Display settings
        self.window_name = "Pump Curve Digitizer"
        self.colors = {
            'pump': (0, 255, 0),  # Green
            'npsh': (0, 0, 255),  # Red
            'calibration': (255, 0, 255),  # Magenta
            'selected': (255, 255, 0),  # Cyan
            'efficiency': (255, 165, 0)  # Orange for efficiency points (on pump curve)
        }

        # Pump metadata: model name, rated speed (rpm), trim allowance, list of impeller options, and current working diameter
        self.pump_model: str = ""
        self.rpm: int = 0  # Rated speed in revolutions per minute
        self.trim_allowed: bool = False
        self.impeller_options: List[Dict] = []
        self.current_diameter: float = 0.0

    def reset_for_new_page(self):
        """
        Reset all curve data to prepare for processing a new page.
        Keeps calibration and pump metadata, but clears digitized curves.
        """
        self.pump_curves.clear()
        self.npsh_curves.clear()
        self.efficiency_points.clear()
        self.impeller_options = []
        self.current_diameter = 0.0
        # Note: pump_model, rpm, trim_allowed are kept as they might be the same,
        # but they will be overwritten by get_pump_info() on the next page.

    def load_pdf_page(self, pdf_path: str, page_num: int = 1, dpi: int = 300):
        """Load a PDF page as image."""
        print(f"📄 Loading page {page_num} from {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
        self.original_image = np.array(images[0])
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
        self.image = self.original_image.copy()
        print(f"✅ Image loaded: {self.image.shape[1]}x{self.image.shape[0]}")

    def select_graph_area(self):
        """Step 1: Select the graph area."""
        print("\n📏 STEP 1: Select Graph Area")
        print("Drag and select the rectangle containing the graph")
        print("Press ENTER when done, ESC to cancel")

        roi = cv2.selectROI(self.window_name, self.image, False)
        cv2.destroyWindow(self.window_name)

        if roi == (0, 0, 0, 0):
            print("❌ Graph area selection cancelled")
            return False

        x, y, w, h = [int(v) for v in roi]
        self.graph_rect = (x, y, w, h)

        # Crop image to graph area for further work
        self.image = self.image[y:y + h, x:x + w]
        print(f"✅ Graph area selected: ({x}, {y}, {w}, {h})")
        return True

    def calibrate_axes(self):
        """Step 2: Calibrate all axes by clicking two points per axis."""
        print("\n📐 STEP 2: Axis Calibration (Two points per axis)")
        print("\nWe need to calibrate axes by clicking two known points for each.")
        print("Order of clicks:")
        print("1. X-axis: first point (left, known flow)")
        print("2. X-axis: second point (right, known flow)")
        print("3. Y-axis HEAD (left): first point (bottom, known head)")
        print("4. Y-axis HEAD (left): second point (top, known head)")
        print("5. Y-axis NPSH (right): first point (bottom, known NPSH) – press 's' to skip if not present")
        print("6. Y-axis NPSH (right): second point (top, known NPSH)")
        print("7. Y-axis EFFICIENCY (right): first point (bottom, known %) – press 's' to skip")
        print("8. Y-axis EFFICIENCY (right): second point (top, known %)")
        print("\nFor each point, you'll enter the real value.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)

        # Store calibration points in order
        calib_points = []  # list of (x_px, y_px, description)
        values = []

        # Expected sequence (some may be skipped)
        expected_points = [
            ("X1", "X-axis first point (left)"),
            ("X2", "X-axis second point (right)"),
            ("Y_HEAD1", "Y-head first point (bottom)"),
            ("Y_HEAD2", "Y-head second point (top)"),
            ("Y_NPSH1", "Y-NPSH first point (bottom)"),
            ("Y_NPSH2", "Y-NPSH second point (top)"),
            ("Y_EFF1", "Y-efficiency first point (bottom)"),
            ("Y_EFF2", "Y-efficiency second point (top)")
        ]

        # Flags for skipped axes
        skip_npsh = False
        skip_eff = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal skip_npsh, skip_eff
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = len(calib_points)
                if idx >= len(expected_points):
                    return

                desc = expected_points[idx][1]
                print(f"\nPoint {idx + 1}: {desc}")

                # Draw point
                img_copy = self.image.copy()
                for i, (px, py, _) in enumerate(calib_points):
                    cv2.circle(img_copy, (px, py), 5, self.colors['calibration'], -1)
                    cv2.putText(img_copy, str(i + 1), (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['calibration'], 2)

                cv2.circle(img_copy, (x, y), 5, self.colors['calibration'], -1)
                cv2.putText(img_copy, str(idx + 1), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['calibration'], 2)
                cv2.imshow(self.window_name, img_copy)

                # Ask for value
                if "first point" in desc and ("NPSH" in desc or "efficiency" in desc):
                    # For NPSH and efficiency first points, ask if they want to skip this axis
                    choice = input(f"  Enter value for {desc} (or 's' to skip this axis): ").strip()
                    if choice.lower() == 's':
                        # Skip this and the next point of this axis
                        if "NPSH" in desc:
                            skip_npsh = True
                        elif "efficiency" in desc:
                            skip_eff = True
                        print(f"  Skipping {desc.split()[0]} axis.")
                        # We still need to record the click but with dummy value? Actually we skip entirely.
                        # To keep indexing consistent, we'll just not add these two points.
                        # We'll handle by not incrementing calib_points for skipped ones.
                        # But the click event already happened; we need to prevent adding.
                        # Let's just return without adding.
                        return
                    else:
                        value = float(choice)
                else:
                    value = float(input(f"  Enter real value: ").strip())

                calib_points.append((x, y, expected_points[idx][0]))
                values.append(value)

        cv2.setMouseCallback(self.window_name, mouse_callback)

        # Show instructions on image
        img_disp = self.image.copy()
        y_pos = 30
        instructions = [
            "Calibration: Click two points per axis",
            "1. X-axis left",
            "2. X-axis right",
            "3. Y-head bottom",
            "4. Y-head top",
            "5. Y-NPSH bottom (s to skip)",
            "6. Y-NPSH top",
            "7. Y-efficiency bottom (s to skip)",
            "8. Y-efficiency top"
        ]
        for text in instructions:
            cv2.putText(img_disp, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['calibration'], 1)
            y_pos += 25

        cv2.imshow(self.window_name, img_disp)

        print("\nClick on the points in order. Press 'c' when done (after all required points), ESC to cancel")

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return False
            elif key == ord('c'):
                # Check we have at least X and Y head points (4 points)
                if len(calib_points) >= 4:
                    break
                else:
                    print("⚠️ Need at least X and Y head points (4 points). Continue clicking.")

        cv2.destroyAllWindows()

        # Build calibration object
        # X points
        x_p1 = next((p for p in calib_points if p[2] == "X1"), None)
        x_p2 = next((p for p in calib_points if p[2] == "X2"), None)
        if not x_p1 or not x_p2:
            print("❌ Missing X-axis calibration points")
            return False

        x_points = (
            CalibrationPoint(x_p1[0], x_p1[1], values[calib_points.index(x_p1)], "X1"),
            CalibrationPoint(x_p2[0], x_p2[1], values[calib_points.index(x_p2)], "X2")
        )

        # Y head points
        yh_p1 = next((p for p in calib_points if p[2] == "Y_HEAD1"), None)
        yh_p2 = next((p for p in calib_points if p[2] == "Y_HEAD2"), None)
        if not yh_p1 or not yh_p2:
            print("❌ Missing Y-head calibration points")
            return False

        y_head_points = (
            CalibrationPoint(yh_p1[0], yh_p1[1], values[calib_points.index(yh_p1)], "Y_HEAD1"),
            CalibrationPoint(yh_p2[0], yh_p2[1], values[calib_points.index(yh_p2)], "Y_HEAD2")
        )

        # Y NPSH points (optional)
        y_npsh_points = None
        if not skip_npsh:
            yn_p1 = next((p for p in calib_points if p[2] == "Y_NPSH1"), None)
            yn_p2 = next((p for p in calib_points if p[2] == "Y_NPSH2"), None)
            if yn_p1 and yn_p2:
                y_npsh_points = (
                    CalibrationPoint(yn_p1[0], yn_p1[1], values[calib_points.index(yn_p1)], "Y_NPSH1"),
                    CalibrationPoint(yn_p2[0], yn_p2[1], values[calib_points.index(yn_p2)], "Y_NPSH2")
                )

        # Y efficiency points (optional)
        y_eff_points = None
        if not skip_eff:
            ye_p1 = next((p for p in calib_points if p[2] == "Y_EFF1"), None)
            ye_p2 = next((p for p in calib_points if p[2] == "Y_EFF2"), None)
            if ye_p1 and ye_p2:
                y_eff_points = (
                    CalibrationPoint(ye_p1[0], ye_p1[1], values[calib_points.index(ye_p1)], "Y_EFF1"),
                    CalibrationPoint(ye_p2[0], ye_p2[1], values[calib_points.index(ye_p2)], "Y_EFF2")
                )

        self.calibration = AxisCalibration(
            x_points=x_points,
            y_head_points=y_head_points,
            y_npsh_points=y_npsh_points,
            y_efficiency_points=y_eff_points
        )

        print("\n✅ Calibration complete!")
        return True

    def px_to_real(self, x_px: int, y_px: int, axis_type: str = 'head') -> Tuple[float, float]:
        """Convert pixel coordinates to real values using two-point calibration."""
        if not self.calibration:
            return (float(x_px), float(y_px))

        # X conversion
        a_x, b_x = self.calibration.get_x_transform()
        x_real = a_x * x_px + b_x

        # Y conversion based on axis type
        if axis_type == 'head':
            a_y, b_y = self.calibration.get_y_head_transform()
            y_real = a_y * y_px + b_y
        elif axis_type == 'npsh':
            trans = self.calibration.get_y_npsh_transform()
            if trans:
                a_y, b_y = trans
                y_real = a_y * y_px + b_y
            else:
                y_real = 0.0
        elif axis_type == 'efficiency':
            trans = self.calibration.get_y_efficiency_transform()
            if trans:
                a_y, b_y = trans
                y_real = a_y * y_px + b_y
            else:
                y_real = 0.0
        else:
            y_real = float(y_px)

        return (x_real, y_real)

    def get_pump_info(self):
        """
        Step: Gather pump metadata from the user.
        Asks for pump model, rated speed (rpm), and whether impeller trimming is allowed.
        This information will be stored and later included in the JSON export.
        """
        print("\nℹ️ PUMP INFORMATION")

        # Request pump model (e.g., "NSC125-80-210")
        self.pump_model = input("Enter pump model (e.g., NSC125-80-210): ").strip()

        # Request rated speed (rpm)
        rpm_str = input("Enter rated speed (rpm) (e.g., 2980, 1450, 980, 740): ").strip()
        try:
            self.rpm = int(rpm_str)
        except ValueError:
            print("⚠️ Invalid number, setting rpm to 0.")
            self.rpm = 0

        # Ask if the impeller can be trimmed to match a duty point
        trim = input("Can impeller be trimmed to match duty point? (y/n): ").strip().lower()
        self.trim_allowed = (trim == 'y')

        # Confirm the entered data
        print(f"✅ Pump model: {self.pump_model}, rpm: {self.rpm}, trim_allowed={self.trim_allowed}")

    def digitize_curves(self):
        """
        Digitize all curves for a single pump with multiple impeller diameters.

        Steps for each diameter:
            - Q-H curve (mandatory)
            - NPSH curve (optional)
            - Efficiency points (optional)
        The user clicks points along the curve; 'c' finishes the current curve.
        All points are stored in the corresponding curve objects.
        """
        print("\n📊 STEP 3: Digitize Curves")
        print("\nNow we will digitize curves for different impeller diameters.")

        num_curves = int(input("How many impeller diameters (curves) are shown? ").strip())

        # Clear any previously stored curves (for a fresh start)
        self.pump_curves.clear()
        self.npsh_curves.clear()
        self.efficiency_points.clear()

        # Create OpenCV window for interactive point selection
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)

        for curve_idx in range(num_curves):
            print(f"\n--- Curve #{curve_idx + 1} ---")
            diam = float(input(f"  Diameter for curve #{curve_idx + 1} (mm): ").strip())
            self.current_diameter = diam

            # --- Digitize Q-H curve ---
            print(f"\n  Now digitize the Q-H curve for diameter {diam} mm.")
            print("  Click points along the curve. Include start and end.")
            print("  Press 'c' when done with this curve.")

            pump_curve = PumpCurve(
                name=f"Pump_{curve_idx + 1}",
                pump_number=curve_idx + 1,
                diameter=diam,
                color=self.colors['pump']
            )
            self.pump_curves.append(pump_curve)

            points_temp = []  # temporary storage for points of this curve

            def mouse_callback_qh(event, x, y, flags, param):
                """Mouse callback for Q-H curve digitization."""
                if event == cv2.EVENT_LBUTTONDOWN:
                    points_temp.append((x, y))
                    # Update display
                    img_disp = self.image.copy()
                    for (px, py) in points_temp:
                        cv2.circle(img_disp, (px, py), 4, self.colors['pump'], -1)
                    if len(points_temp) > 1:
                        pts = np.array(points_temp, np.int32)
                        cv2.polylines(img_disp, [pts], False, self.colors['pump'], 2)
                    cv2.putText(img_disp, f"Diameter {diam} mm - Q-H", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['pump'], 2)
                    cv2.putText(img_disp, f"Points: {len(points_temp)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow(self.window_name, img_disp)

            cv2.setMouseCallback(self.window_name, mouse_callback_qh)

            # Initial display with instructions
            img_disp = self.image.copy()
            cv2.putText(img_disp, f"Diameter {diam} mm - Q-H", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['pump'], 2)
            cv2.putText(img_disp, "Click points, press 'c' when done", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(self.window_name, img_disp)

            # Wait for user to finish this curve
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('c'):
                    break

            # Store collected points into the curve object
            for (x, y) in points_temp:
                pump_curve.add_point(x, y)

            # --- Optionally digitize NPSH curve ---
            has_npsh = input(f"\n  Does this diameter ({diam} mm) have an NPSH curve? (y/n): ").strip().lower() == 'y'
            if has_npsh:
                print(f"  Now digitize the NPSH curve for diameter {diam} mm.")
                print("  Click points, press 'c' when done.")

                npsh_curve = NPSHCurve(
                    name=f"NPSH_{curve_idx + 1}",
                    pump_number=curve_idx + 1,
                    diameter=diam,
                    color=self.colors['npsh']
                )
                self.npsh_curves.append(npsh_curve)

                points_npsh = []

                def mouse_callback_npsh(event, x, y, flags, param):
                    """Mouse callback for NPSH curve."""
                    if event == cv2.EVENT_LBUTTONDOWN:
                        points_npsh.append((x, y))
                        img_disp = self.image.copy()
                        for (px, py) in points_npsh:
                            cv2.circle(img_disp, (px, py), 4, self.colors['npsh'], -1)
                        if len(points_npsh) > 1:
                            pts = np.array(points_npsh, np.int32)
                            cv2.polylines(img_disp, [pts], False, self.colors['npsh'], 2)
                        cv2.putText(img_disp, f"Diameter {diam} mm - NPSH", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['npsh'], 2)
                        cv2.imshow(self.window_name, img_disp)

                cv2.setMouseCallback(self.window_name, mouse_callback_npsh)

                img_disp = self.image.copy()
                cv2.putText(img_disp, f"Diameter {diam} mm - NPSH", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['npsh'], 2)
                cv2.putText(img_disp, "Click points, press 'c' when done", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow(self.window_name, img_disp)

                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('c'):
                        break

                for (x, y) in points_npsh:
                    npsh_curve.add_point(x, y)

            # --- Optionally digitize efficiency points ---
            has_eff = input(f"\n  Are there efficiency points for diameter {diam} mm? (y/n): ").strip().lower() == 'y'
            if has_eff:
                print(f"  Now click on the Q-H curve at points where efficiency is known.")
                print("  For each click, enter the efficiency value.")

                points_eff = []

                def mouse_callback_eff(event, x, y, flags, param):
                    """Mouse callback for efficiency points (click on Q-H curve)."""
                    if event == cv2.EVENT_LBUTTONDOWN:
                        val = input(f"    Efficiency at this point (%): ").strip()
                        try:
                            eff_val = float(val)
                            points_eff.append((x, y, eff_val))
                            # Update display
                            img_disp = self.image.copy()
                            # Show Q-H curve in background
                            for pt in pump_curve.points:
                                cv2.circle(img_disp, (pt.x_px, pt.y_px), 4, self.colors['pump'], -1)
                            if len(pump_curve.points) > 1:
                                pts = np.array([[p.x_px, p.y_px] for p in pump_curve.points], np.int32)
                                cv2.polylines(img_disp, [pts], False, self.colors['pump'], 2)
                            # Show efficiency points
                            for (px, py, ev) in points_eff:
                                cv2.circle(img_disp, (px, py), 6, self.colors['efficiency'], 2)
                                cv2.putText(img_disp, f"{ev}%", (px + 10, py - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['efficiency'], 1)
                            cv2.putText(img_disp, f"Diameter {diam} mm - Efficiency points", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['efficiency'], 2)
                            cv2.imshow(self.window_name, img_disp)
                        except ValueError:
                            print("    Invalid number, skipping.")

                cv2.setMouseCallback(self.window_name, mouse_callback_eff)

                # Show the Q-H curve as a guide
                img_disp = self.image.copy()
                for pt in pump_curve.points:
                    cv2.circle(img_disp, (pt.x_px, pt.y_px), 4, self.colors['pump'], -1)
                if len(pump_curve.points) > 1:
                    pts = np.array([[p.x_px, p.y_px] for p in pump_curve.points], np.int32)
                    cv2.polylines(img_disp, [pts], False, self.colors['pump'], 2)
                cv2.putText(img_disp, f"Diameter {diam} mm - Click on Q-H curve for efficiency", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['efficiency'], 2)
                cv2.putText(img_disp, "Press 'c' when done", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow(self.window_name, img_disp)

                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('c'):
                        break

                # Store efficiency points
                for (x, y, val) in points_eff:
                    self.efficiency_points.append(
                        EfficiencyPoint(
                            pump_curve_name=pump_curve.name,
                            pump_number=curve_idx + 1,
                            diameter=diam,
                            x_px=x,
                            y_px=y,
                            efficiency_value=val
                        )
                    )

            # Reset mouse callback after finishing this diameter
            cv2.setMouseCallback(self.window_name, lambda *args: None)

        cv2.destroyAllWindows()
        print("\n✅ Digitization complete!")
        return True

    def _update_display(self, all_curves, current_idx):
        """Update the display with all curves drawn so far."""
        img_disp = self.image.copy()

        # Draw all curves up to current
        for i, curve in enumerate(all_curves[:current_idx + 1]):
            color = curve.color
            points = curve.points

            # Draw points
            for pt in points:
                cv2.circle(img_disp, (pt.x_px, pt.y_px), 4, color, -1)
                # If efficiency point, add a small orange dot
                if pt.efficiency is not None:
                    cv2.circle(img_disp, (pt.x_px, pt.y_px), 6, self.colors['efficiency'], 1)

            # Draw polyline
            if len(points) > 1:
                pts = np.array([[p.x_px, p.y_px] for p in points], np.int32)
                cv2.polylines(img_disp, [pts], False, color, 2)

        # Instructions
        y_pos = 30
        if current_idx < len(all_curves):
            cv2.putText(img_disp, f"Digitizing: {all_curves[current_idx].name}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['selected'], 2)
            y_pos += 30
        cv2.putText(img_disp, "Click to add points", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
        cv2.putText(img_disp, "'n' next curve, 'c' done", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(self.window_name, img_disp)

    def fit_curves(self) -> List[FittedCurve]:
        """
        Fit cubic polynomials to the digitized points for each impeller diameter.
        Groups curves by diameter, fits Q-H and NPSH separately, and collects efficiency points.
        Efficiency points are fitted with a 4th-degree polynomial.
        Returns a list of FittedCurve objects (one per curve).
        """
        print("\n📈 Fitting curves...")
        fitted_curves = []

        # Group curves by diameter
        curves_by_diam = {}
        for pump in self.pump_curves:
            curves_by_diam[pump.diameter] = {'pump': pump, 'npsh': None}
        for npsh in self.npsh_curves:
            if npsh.diameter in curves_by_diam:
                curves_by_diam[npsh.diameter]['npsh'] = npsh
            else:
                curves_by_diam[npsh.diameter] = {'pump': None, 'npsh': npsh}

        for diam, curves in curves_by_diam.items():
            pump_curve = curves['pump']
            npsh_curve = curves['npsh']

            # --- Fit Q-H curve ---
            if pump_curve and len(pump_curve.points) >= 4:
                # ... (existing code to convert points, fit cubic) ...
                points_real = []
                for p in pump_curve.points:
                    xr, yr = self.px_to_real(p.x_px, p.y_px, 'head')
                    points_real.append((xr, yr))
                points_real.sort(key=lambda p: p[0])
                x_real = np.array([p[0] for p in points_real])
                y_real = np.array([p[1] for p in points_real])

                coeffs = np.polyfit(x_real, y_real, 3)
                poly = np.poly1d(coeffs)
                y_fitted = poly(x_real)
                rmse = np.sqrt(np.mean((y_real - y_fitted) ** 2))
                x_min, x_max = x_real[0], x_real[-1]
                y_range = (float(min(y_real)), float(max(y_real)))
                y_span = y_range[1] - y_range[0]
                confidence = 1.0 - min(1.0, rmse / (y_span if y_span > 0 else 1))

                # Collect efficiency points for this diameter
                eff_points = []
                for ep in self.efficiency_points:
                    if ep.diameter == diam:
                        xr, _ = self.px_to_real(ep.x_px, ep.y_px, 'head')
                        eff_points.append((xr, ep.efficiency_value))

                # Initialize efficiency fitting results
                efficiency_coeffs = None
                efficiency_rmse = None
                efficiency_x_min = None
                efficiency_x_max = None
                if len(eff_points) >= 5:  # Need at least 5 points for 4th degree polynomial
                    # Sort efficiency points by flow
                    eff_points.sort(key=lambda p: p[0])
                    x_eff = np.array([p[0] for p in eff_points])
                    y_eff = np.array([p[1] for p in eff_points])
                    try:
                        # Fit 4th degree polynomial
                        eff_coeffs = np.polyfit(x_eff, y_eff, 4)
                        eff_poly = np.poly1d(eff_coeffs)
                        y_eff_fitted = eff_poly(x_eff)
                        eff_rmse = np.sqrt(np.mean((y_eff - y_eff_fitted) ** 2))
                        efficiency_coeffs = eff_coeffs.tolist()
                        efficiency_rmse = float(eff_rmse)
                        efficiency_x_min = float(x_eff[0])
                        efficiency_x_max = float(x_eff[-1])
                    except Exception as e:
                        print(f"  ⚠️ Could not fit efficiency for d={diam}: {e}")

                fitted = FittedCurve(
                    name=f"Pump_d{diam}",
                    curve_type='pump',
                    points_px=[(p.x_px, p.y_px) for p in pump_curve.points],
                    points_real=points_real,
                    coefficients=coeffs.tolist(),
                    x_range=(float(x_min), float(x_max)),
                    y_range=y_range,
                    rmse=float(rmse),
                    confidence=float(confidence),
                    efficiency_points=eff_points,
                    efficiency_coefficients=efficiency_coeffs,
                    efficiency_rmse=efficiency_rmse,
                    efficiency_x_min=efficiency_x_min,
                    efficiency_x_max=efficiency_x_max
                )
                fitted_curves.append(fitted)
                print(f"  ✅ Pump d={diam}: RMSE={rmse:.4f}, Confidence={confidence:.2f}, Eff points: {len(eff_points)}")
                if efficiency_coeffs:
                    print(
                        f"      Efficiency fitted, RMSE={efficiency_rmse:.4f}, range [{efficiency_x_min:.2f}, {efficiency_x_max:.2f}]")

            # --- Fit NPSH curve ---
            if npsh_curve and len(npsh_curve.points) >= 4:
                points_real = []
                for p in npsh_curve.points:
                    xr, yr = self.px_to_real(p.x_px, p.y_px, 'npsh')
                    points_real.append((xr, yr))
                points_real.sort(key=lambda p: p[0])
                x_real = np.array([p[0] for p in points_real])
                y_real = np.array([p[1] for p in points_real])

                coeffs = np.polyfit(x_real, y_real, 3)
                poly = np.poly1d(coeffs)
                y_fitted = poly(x_real)
                rmse = np.sqrt(np.mean((y_real - y_fitted) ** 2))
                x_min, x_max = x_real[0], x_real[-1]
                y_range = (float(min(y_real)), float(max(y_real)))
                y_span = y_range[1] - y_range[0]
                confidence = 1.0 - min(1.0, rmse / (y_span if y_span > 0 else 1))

                fitted = FittedCurve(
                    name=f"NPSH_d{diam}",
                    curve_type='npsh',
                    points_px=[(p.x_px, p.y_px) for p in npsh_curve.points],
                    points_real=points_real,
                    coefficients=coeffs.tolist(),
                    x_range=(float(x_min), float(x_max)),
                    y_range=y_range,
                    rmse=float(rmse),
                    confidence=float(confidence)
                )
                fitted_curves.append(fitted)
                print(f"  ✅ NPSH d={diam}: RMSE={rmse:.4f}, Confidence={confidence:.2f}")

        return fitted_curves

    def export_results(self, fitted_curves: List[FittedCurve], output_prefix: str = "pump_data"):
        """Export results in multiple formats."""

        # 1. Export equations to CSV
        equations_data = []
        for curve in fitted_curves:
            a, b, c, d = curve.coefficients
            equations_data.append({
                'curve_name': curve.name,
                'type': curve.curve_type,
                'a_x3': a,
                'b_x2': b,
                'c_x1': c,
                'd_const': d,
                'x_min': curve.x_range[0],
                'x_max': curve.x_range[1],
                'y_min': curve.y_range[0],
                'y_max': curve.y_range[1],
                'rmse': curve.rmse,
                'confidence': curve.confidence
            })

        df_eq = pd.DataFrame(equations_data)
        df_eq.to_csv(f"{output_prefix}_equations.csv", index=False)
        print(f"✅ Equations saved to {output_prefix}_equations.csv")

        # 2. Export 12 equally spaced points for each curve
        points_data = []
        for curve in fitted_curves:
            x_values = np.linspace(curve.x_range[0], curve.x_range[1], 12)
            poly = np.poly1d(curve.coefficients)
            y_values = poly(x_values)

            for i, (x, y) in enumerate(zip(x_values, y_values)):
                points_data.append({
                    'curve_name': curve.name,
                    'type': curve.curve_type,
                    'point_num': i + 1,
                    'x': x,
                    'y': y
                })

        df_points = pd.DataFrame(points_data)
        df_points.to_csv(f"{output_prefix}_points.csv", index=False)
        print(f"✅ 12-point samples saved to {output_prefix}_points.csv")

        # 3. Export efficiency points
        if self.efficiency_points:
            eff_data = []
            for ep in self.efficiency_points:
                xr, _ = self.px_to_real(ep.x_px, ep.y_px, 'head')
                eff_data.append({
                    'pump': ep.pump_curve_name,
                    'pump_number': ep.pump_number,
                    'flow': xr,
                    'efficiency': ep.efficiency_value
                })
            df_eff = pd.DataFrame(eff_data)
            df_eff.to_csv(f"{output_prefix}_efficiency.csv", index=False)
            print(f"✅ Efficiency points saved to {output_prefix}_efficiency.csv")

        # 4. Export Python functions
        with open(f"{output_prefix}_functions.py", "w") as f:
            f.write("# Pump Curve Functions\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")

            for curve in fitted_curves:
                a, b, c, d = curve.coefficients
                f.write(f"""
def {curve.name.lower().replace('-', '_').replace(' ', '_')}(x):
    \"\"\"
    {curve.name} curve ({curve.curve_type})
    Valid for x in [{curve.x_range[0]:.2f}, {curve.x_range[1]:.2f}]
    RMSE: {curve.rmse:.4f}
    \"\"\"
    return {a:.6f}*x**3 + {b:.6f}*x**2 + {c:.6f}*x + {d:.6f}
""")

        print(f"✅ Python functions saved to {output_prefix}_functions.py")

    def export_to_json(self, fitted_curves: List[FittedCurve], filename: str = "pump_catalog.json"):
        """
        Export the digitized pump data to a JSON file, appending to existing catalog.
        Merges new data with existing entries: if same pump model AND same rpm exist,
        adds only new impeller diameters.
        Structure:
        {
            "pumps": [
                {
                    "model": "...",
                    "rpm": 2980,
                    "trim_allowed": true/false,
                    "impeller_options": [
                        {
                            "diameter": 210,
                            "curves": {
                                "q_h": {"coefficients": [...], "q_min": ..., "q_max": ...},
                                "npsh": {...}   (optional)
                            },
                            "efficiency": {
                                "coefficients": [...],
                                "rmse": ...,
                                "q_min": ...,   # from efficiency points
                                "q_max": ...    # from efficiency points
                            }
                        },
                        ...
                    ]
                }
            ]
        }
        """

        # Group fitted curves by diameter
        diam_dict = {}
        for curve in fitted_curves:
            # Extract diameter from curve name (e.g., "Pump_d210" or "NPSH_d210")
            match = re.search(r'_d(\d+)', curve.name)
            if not match:
                continue
            diam = float(match.group(1))
            if diam not in diam_dict:
                diam_dict[diam] = {'pump': None, 'npsh': None}
            if curve.curve_type == 'pump':
                diam_dict[diam]['pump'] = curve
            elif curve.curve_type == 'npsh':
                diam_dict[diam]['npsh'] = curve

        # Build new impeller options list
        new_impeller_options = []
        for diam, curves in diam_dict.items():
            pump_curve = curves['pump']
            npsh_curve = curves['npsh']
            if pump_curve is None:
                continue  # should not happen

            option = {
                'diameter': diam,
                'curves': {}
            }

            # Q-H curve
            option['curves']['q_h'] = {
                'coefficients': pump_curve.coefficients,
                'q_min': pump_curve.x_range[0],
                'q_max': pump_curve.x_range[1]
            }

            # Fitted efficiency curve (if available)
            if pump_curve.efficiency_coefficients:
                option['efficiency'] = {
                    'coefficients': pump_curve.efficiency_coefficients,
                    'rmse': pump_curve.efficiency_rmse,
                    # Use the efficiency-specific x‑range (from user‑clicked points)
                    'q_min': pump_curve.efficiency_x_min,
                    'q_max': pump_curve.efficiency_x_max
                }

            # NPSH curve (if present)
            if npsh_curve:
                option['curves']['npsh'] = {
                    'coefficients': npsh_curve.coefficients,
                    'q_min': npsh_curve.x_range[0],
                    'q_max': npsh_curve.x_range[1]
                }

            new_impeller_options.append(option)

        # New pump entry (includes rpm)
        new_pump = {
            'model': self.pump_model,
            'rpm': self.rpm,
            'trim_allowed': self.trim_allowed,
            'impeller_options': new_impeller_options
        }

        # Read existing catalog if it exists
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    catalog = json.load(f)
                if 'pumps' not in catalog:
                    catalog['pumps'] = []
            except:
                catalog = {'pumps': []}
        else:
            catalog = {'pumps': []}

        # Check if pump with same model AND same rpm already exists
        existing_pump = None
        for pump in catalog['pumps']:
            if pump.get('model') == self.pump_model and pump.get('rpm') == self.rpm:
                existing_pump = pump
                break

        if existing_pump is None:
            # New pump (model+rpm combination), simply append
            catalog['pumps'].append(new_pump)
            print(f"✅ New pump entry added: '{self.pump_model}' at {self.rpm} rpm.")
        else:
            # Merge impeller options: add diameters not already present
            existing_diams = {opt['diameter'] for opt in existing_pump['impeller_options']}
            added_count = 0
            for new_opt in new_impeller_options:
                if new_opt['diameter'] not in existing_diams:
                    existing_pump['impeller_options'].append(new_opt)
                    added_count += 1
            if added_count > 0:
                print(
                    f"✅ Added {added_count} new impeller diameter(s) to existing pump '{self.pump_model}' at {self.rpm} rpm.")
            else:
                print(
                    f"ℹ️ All impeller diameters already exist for pump '{self.pump_model}' at {self.rpm} rpm – no changes made.")

        # Write back
        with open(filename, 'w') as f:
            json.dump(catalog, f, indent=2)
        print(f"✅ JSON catalog saved to {filename} (total pump entries: {len(catalog['pumps'])}")

    def plot_results(self, fitted_curves: List[FittedCurve]):
        """
        Plot original points and fitted curves.
        For pump curves, if efficiency coefficients are available, plot the fitted efficiency curve
        using the efficiency‑specific x‑range (from the actual clicked points).
        """
        plt.figure(figsize=(14, 10))

        colors = {'pump': 'blue', 'npsh': 'red'}

        for curve in fitted_curves:
            if curve.curve_type == 'pump':
                # Original Q-H points
                x_orig = [p[0] for p in curve.points_real]
                y_orig = [p[1] for p in curve.points_real]

                # Fitted Q-H curve (100 points over full Q-H range)
                x_fit_qh = np.linspace(curve.x_range[0], curve.x_range[1], 100)
                poly_qh = np.poly1d(curve.coefficients)
                y_fit_qh = poly_qh(x_fit_qh)

                # Plot Q-H
                plt.scatter(x_orig, y_orig, c=colors['pump'], s=20, alpha=0.5, marker='o',
                            label=f"{curve.name} (Q-H points)")
                plt.plot(x_fit_qh, y_fit_qh, color=colors['pump'], linewidth=2,
                         label=f"{curve.name} (Q-H fit)")

                # If efficiency coefficients exist, plot efficiency curve over its own valid range
                if curve.efficiency_coefficients:
                    poly_eff = np.poly1d(curve.efficiency_coefficients)

                    # Use efficiency‑specific x‑range if available, otherwise fall back to Q‑H range
                    if curve.efficiency_x_min is not None and curve.efficiency_x_max is not None:
                        x_min_eff = curve.efficiency_x_min
                        x_max_eff = curve.efficiency_x_max
                    else:
                        # Fallback (should not happen, but just in case)
                        x_min_eff = curve.x_range[0]
                        x_max_eff = curve.x_range[1]

                    x_fit_eff = np.linspace(x_min_eff, x_max_eff, 100)
                    y_fit_eff = poly_eff(x_fit_eff)

                    plt.plot(x_fit_eff, y_fit_eff, color='orange', linestyle='--', linewidth=2,
                             label=f"{curve.name} Efficiency (fit)")

                    # Plot original efficiency points (if any)
                    if curve.efficiency_points:
                        x_eff = [p[0] for p in curve.efficiency_points]
                        y_eff = [p[1] for p in curve.efficiency_points]
                        plt.scatter(x_eff, y_eff, c='orange', s=40, marker='s', alpha=0.7,
                                    label=f"{curve.name} Efficiency points")

            elif curve.curve_type == 'npsh':
                # Original NPSH points
                x_orig = [p[0] for p in curve.points_real]
                y_orig = [p[1] for p in curve.points_real]

                # Fitted NPSH curve
                x_fit = np.linspace(curve.x_range[0], curve.x_range[1], 100)
                poly = np.poly1d(curve.coefficients)
                y_fit = poly(x_fit)

                plt.scatter(x_orig, y_orig, c=colors['npsh'], s=20, alpha=0.5, marker='^',
                            label=f"{curve.name} (points)")
                plt.plot(x_fit, y_fit, color=colors['npsh'], linewidth=2,
                         label=f"{curve.name} (fit)")

        plt.xlabel('Flow Rate')
        plt.ylabel('Head / NPSH')
        plt.title('Digitized Pump Curves with Polynomial Fit')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def run(self, pdf_path: str):
        """
        Main execution pipeline with multi‑page support.
        User can process multiple pages sequentially, each time adding data to the JSON catalog.
        """
        print("\n" + "=" * 60)
        print("PROFESSIONAL PUMP CURVE DIGITIZER")
        print("=" * 60)

        # Loop for processing multiple pages
        page_counter = 0
        while True:
            page_counter += 1
            print(f"\n{'=' * 60}")
            print(f"PROCESSING PAGE {page_counter}")
            print(f"{'=' * 60}")

            # Ask which page to process (first page may have default)
            if page_counter == 1:
                page_input = input("\nEnter page number to process (default 1): ").strip()
                current_page = int(page_input) if page_input else 1
            else:
                current_page = int(input(f"Enter next page number (or 0 to quit): ").strip())
                if current_page <= 0:
                    break

            # Load the PDF page
            self.load_pdf_page(pdf_path, page_num=current_page, dpi=300)

            # Step 2: Select graph area
            if not self.select_graph_area():
                print("⚠️ Graph area selection failed. Skipping this page.")
                continue

            # Step 3: Calibrate axes (re‑calibrate each page – could be refined later)
            if not self.calibrate_axes():
                print("⚠️ Calibration failed. Skipping this page.")
                continue

            # Step 4: Get pump information (model, rpm, trim)
            self.get_pump_info()

            # Step 5: Digitize curves
            if not self.digitize_curves():
                print("⚠️ Digitization failed. Skipping this page.")
                continue

            # Step 6: Fit curves
            fitted_curves = self.fit_curves()
            if not fitted_curves:
                print("❌ No curves could be fitted. Skipping this page.")
                continue

            # Step 7: Export results for this page
            print("\n📁 Export results for this page")
            print("Choose format:")
            print("1. CSV (equations, points, efficiency)")
            print("2. JSON (for Java application/database)")
            print("3. Both")
            choice = input("Enter 1, 2, or 3: ").strip()

            if choice in ['1', '3']:
                self.export_results(fitted_curves)  # CSV export (overwrites files each time)
            if choice in ['2', '3']:
                self.export_to_json(fitted_curves)  # Appends/merges into catalog

            # Step 8: Show plot (optional)
            show_plot = input("\nShow plot for this page? (y/n): ").strip().lower()
            if show_plot == 'y':
                self.plot_results(fitted_curves)

            # Ask if user wants to process another page
            again = input("\nProcess another page? (y/n): ").strip().lower()
            if again != 'y':
                break

            # Reset for the next page (clear curves but keep nothing)
            self.reset_for_new_page()
            # The next loop iteration will load the new page and repeat

        print(f"\n✅ All done! Processed {page_counter} page(s).")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    import sys

    if len(sys.argv) == 2:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter PDF file path: ").strip()
        pdf_path = pdf_path.strip('"').strip("'")

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    digitizer = PumpCurveDigitizer()
    digitizer.run(pdf_path)


if __name__ == "__main__":
    main()