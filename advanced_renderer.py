"""
Advanced Motion Rendering Engine for HTML to GIF Conversion

This module provides advanced motion rendering capabilities including:
- Parametric motion equations
- Bezier curve interpolation
- Spiral animations
- Complex motion compositing
"""

import numpy as np
from typing import Tuple, List, Callable, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math


class MotionType(Enum):
    """Enumeration of available motion types"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    PARAMETRIC = "parametric"
    BEZIER = "bezier"
    SPIRAL = "spiral"
    CIRCULAR = "circular"
    LISSAJOUS = "lissajous"


@dataclass
class Point2D:
    """Represents a 2D point"""
    x: float
    y: float

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x / scalar, self.y / scalar)

    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (x, y)"""
        return (self.x, self.y)


@dataclass
class BezierControlPoints:
    """Represents control points for a cubic Bezier curve"""
    p0: Point2D  # Start point
    p1: Point2D  # First control point
    p2: Point2D  # Second control point
    p3: Point2D  # End point

    def evaluate(self, t: float) -> Point2D:
        """
        Evaluate Bezier curve at parameter t (0 <= t <= 1)
        Uses De Casteljau's algorithm for numerical stability
        """
        t = max(0.0, min(1.0, t))
        mt = 1.0 - t

        # Bernstein polynomials
        b0 = mt ** 3
        b1 = 3.0 * mt ** 2 * t
        b2 = 3.0 * mt * t ** 2
        b3 = t ** 3

        # Compute point on curve
        x = b0 * self.p0.x + b1 * self.p1.x + b2 * self.p2.x + b3 * self.p3.x
        y = b0 * self.p0.y + b1 * self.p1.y + b2 * self.p2.y + b3 * self.p3.y

        return Point2D(x, y)

    def derivative(self, t: float) -> Point2D:
        """Calculate the derivative (tangent) of the Bezier curve at parameter t"""
        t = max(0.0, min(1.0, t))
        mt = 1.0 - t

        # Derivative of Bernstein polynomials
        db0 = -3.0 * mt ** 2
        db1 = 3.0 * mt ** 2 - 6.0 * mt * t
        db2 = 6.0 * mt * t - 3.0 * t ** 2
        db3 = 3.0 * t ** 2

        # Compute derivative
        dx = db0 * self.p0.x + db1 * self.p1.x + db2 * self.p2.x + db3 * self.p3.x
        dy = db0 * self.p0.y + db1 * self.p1.y + db2 * self.p2.y + db3 * self.p3.y

        return Point2D(dx, dy)

    def arc_length(self, num_segments: int = 100) -> float:
        """Calculate approximate arc length using numerical integration"""
        length = 0.0
        prev_point = self.evaluate(0.0)

        for i in range(1, num_segments + 1):
            t = i / num_segments
            curr_point = self.evaluate(t)
            length += prev_point.distance_to(curr_point)
            prev_point = curr_point

        return length


class ParametricMotion:
    """Parametric motion using mathematical equations"""

    @staticmethod
    def linear(t: float, start: Point2D, end: Point2D) -> Point2D:
        """Linear interpolation"""
        return start * (1.0 - t) + end * t

    @staticmethod
    def ease_in(t: float, start: Point2D, end: Point2D, power: float = 2.0) -> Point2D:
        """Ease-in motion (accelerating)"""
        t_eased = t ** power
        return start * (1.0 - t_eased) + end * t_eased

    @staticmethod
    def ease_out(t: float, start: Point2D, end: Point2D, power: float = 2.0) -> Point2D:
        """Ease-out motion (decelerating)"""
        t_eased = 1.0 - (1.0 - t) ** power
        return start * (1.0 - t_eased) + end * t_eased

    @staticmethod
    def ease_in_out(t: float, start: Point2D, end: Point2D, power: float = 2.0) -> Point2D:
        """Ease-in-out motion (accelerate then decelerate)"""
        if t < 0.5:
            t_eased = 2.0 ** (power - 1) * t ** power
        else:
            t_eased = 1.0 - ((-2.0 * t + 2.0) ** power) / 2.0
        return start * (1.0 - t_eased) + end * t_eased

    @staticmethod
    def circular_motion(t: float, center: Point2D, radius: float, 
                       start_angle: float = 0.0) -> Point2D:
        """Circular motion around a center point"""
        angle = start_angle + 2.0 * math.pi * t
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        return Point2D(x, y)

    @staticmethod
    def lissajous_motion(t: float, center: Point2D, width: float, height: float,
                        freq_x: float = 3.0, freq_y: float = 2.0, 
                        phase: float = 0.0) -> Point2D:
        """Lissajous curve motion"""
        x = center.x + width * math.sin(freq_x * 2.0 * math.pi * t + phase)
        y = center.y + height * math.sin(freq_y * 2.0 * math.pi * t)
        return Point2D(x, y)


class SpiralAnimation:
    """Advanced spiral animation engine"""

    @staticmethod
    def archimedean_spiral(t: float, center: Point2D, growth_rate: float = 0.5,
                          start_angle: float = 0.0, max_radius: float = 200.0) -> Point2D:
        """
        Archimedean spiral: r = a + b*θ
        
        Args:
            t: Parameter from 0 to 1
            center: Center point of spiral
            growth_rate: Rate of spiral growth (b parameter)
            start_angle: Starting angle in radians
            max_radius: Maximum radius of spiral
        """
        theta = start_angle + 4.0 * math.pi * t
        radius = growth_rate * theta
        radius = min(radius, max_radius)

        x = center.x + radius * math.cos(theta)
        y = center.y + radius * math.sin(theta)
        return Point2D(x, y)

    @staticmethod
    def logarithmic_spiral(t: float, center: Point2D, growth_rate: float = 0.1,
                          start_angle: float = 0.0) -> Point2D:
        """
        Logarithmic spiral: r = a * e^(b*θ)
        
        Args:
            t: Parameter from 0 to 1
            center: Center point of spiral
            growth_rate: Exponential growth rate
            start_angle: Starting angle in radians
        """
        theta = start_angle + 4.0 * math.pi * t
        radius = math.exp(growth_rate * theta / (2.0 * math.pi))

        x = center.x + radius * math.cos(theta)
        y = center.y + radius * math.sin(theta)
        return Point2D(x, y)

    @staticmethod
    def fermat_spiral(t: float, center: Point2D, scale: float = 50.0,
                     start_angle: float = 0.0) -> Point2D:
        """
        Fermat spiral (parabolic spiral): r = √θ
        
        Args:
            t: Parameter from 0 to 1
            center: Center point of spiral
            scale: Scale factor for radius
            start_angle: Starting angle in radians
        """
        theta = start_angle + 8.0 * math.pi * t
        radius = scale * math.sqrt(abs(theta) / (2.0 * math.pi))

        x = center.x + radius * math.cos(theta)
        y = center.y + radius * math.sin(theta)
        return Point2D(x, y)

    @staticmethod
    def hyperbolic_spiral(t: float, center: Point2D, scale: float = 200.0,
                         start_angle: float = 0.0) -> Point2D:
        """
        Hyperbolic spiral: r*θ = a
        
        Args:
            t: Parameter from 0 to 1
            center: Center point of spiral
            scale: Scale factor
            start_angle: Starting angle in radians
        """
        theta = start_angle + 4.0 * math.pi * t + 0.1  # Avoid division by zero
        radius = scale / theta

        x = center.x + radius * math.cos(theta)
        y = center.y + radius * math.sin(theta)
        return Point2D(x, y)


class AdvancedRenderer:
    """Advanced motion rendering engine combining multiple techniques"""

    def __init__(self, width: int, height: int, fps: int = 30):
        """
        Initialize the advanced renderer
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            fps: Frames per second for animation
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.motion_type = MotionType.LINEAR
        self.parametric_func: Optional[Callable] = None
        self.bezier_curves: List[BezierControlPoints] = []

    def set_motion_type(self, motion_type: MotionType):
        """Set the motion type for rendering"""
        self.motion_type = motion_type

    def set_parametric_function(self, func: Callable[[float], Point2D]):
        """Set a custom parametric function for motion"""
        self.parametric_func = func

    def add_bezier_curve(self, control_points: BezierControlPoints):
        """Add a Bezier curve to the rendering pipeline"""
        self.bezier_curves.append(control_points)

    def get_position(self, t: float, start: Point2D, end: Point2D,
                    **kwargs) -> Point2D:
        """
        Get the position of an object at parameter t
        
        Args:
            t: Parameter from 0 to 1
            start: Starting position
            end: Ending position
            **kwargs: Additional parameters for specific motion types
        
        Returns:
            Point2D representing the position
        """
        if self.motion_type == MotionType.LINEAR:
            return ParametricMotion.linear(t, start, end)

        elif self.motion_type == MotionType.EASE_IN:
            power = kwargs.get('power', 2.0)
            return ParametricMotion.ease_in(t, start, end, power)

        elif self.motion_type == MotionType.EASE_OUT:
            power = kwargs.get('power', 2.0)
            return ParametricMotion.ease_out(t, start, end, power)

        elif self.motion_type == MotionType.EASE_IN_OUT:
            power = kwargs.get('power', 2.0)
            return ParametricMotion.ease_in_out(t, start, end, power)

        elif self.motion_type == MotionType.BEZIER and self.bezier_curves:
            curve = self.bezier_curves[0]
            return curve.evaluate(t)

        elif self.motion_type == MotionType.CIRCULAR:
            center = kwargs.get('center', start)
            radius = kwargs.get('radius', start.distance_to(end))
            start_angle = kwargs.get('start_angle', 0.0)
            return ParametricMotion.circular_motion(t, center, radius, start_angle)

        elif self.motion_type == MotionType.LISSAJOUS:
            center = kwargs.get('center', start)
            width = kwargs.get('width', abs(end.x - start.x))
            height = kwargs.get('height', abs(end.y - start.y))
            return ParametricMotion.lissajous_motion(t, center, width, height)

        elif self.motion_type == MotionType.SPIRAL:
            center = kwargs.get('center', start)
            spiral_type = kwargs.get('spiral_type', 'archimedean')
            
            if spiral_type == 'archimedean':
                return SpiralAnimation.archimedean_spiral(
                    t, center,
                    growth_rate=kwargs.get('growth_rate', 0.5),
                    start_angle=kwargs.get('start_angle', 0.0),
                    max_radius=kwargs.get('max_radius', 200.0)
                )
            elif spiral_type == 'logarithmic':
                return SpiralAnimation.logarithmic_spiral(
                    t, center,
                    growth_rate=kwargs.get('growth_rate', 0.1),
                    start_angle=kwargs.get('start_angle', 0.0)
                )
            elif spiral_type == 'fermat':
                return SpiralAnimation.fermat_spiral(
                    t, center,
                    scale=kwargs.get('scale', 50.0),
                    start_angle=kwargs.get('start_angle', 0.0)
                )
            elif spiral_type == 'hyperbolic':
                return SpiralAnimation.hyperbolic_spiral(
                    t, center,
                    scale=kwargs.get('scale', 200.0),
                    start_angle=kwargs.get('start_angle', 0.0)
                )

        elif self.motion_type == MotionType.PARAMETRIC and self.parametric_func:
            return self.parametric_func(t)

        # Default to linear
        return ParametricMotion.linear(t, start, end)

    def generate_keyframes(self, start: Point2D, end: Point2D, 
                          duration_frames: int, **kwargs) -> List[Point2D]:
        """
        Generate a list of keyframes for the animation
        
        Args:
            start: Starting position
            end: Ending position
            duration_frames: Number of frames for the animation
            **kwargs: Additional motion parameters
        
        Returns:
            List of Point2D positions for each frame
        """
        keyframes = []
        for frame in range(duration_frames):
            t = frame / (duration_frames - 1) if duration_frames > 1 else 0.0
            position = self.get_position(t, start, end, **kwargs)
            keyframes.append(position)
        return keyframes

    def generate_bezier_path(self, control_points: BezierControlPoints,
                            num_points: int = 100) -> List[Point2D]:
        """
        Generate a path along a Bezier curve
        
        Args:
            control_points: BezierControlPoints object
            num_points: Number of points to sample along the curve
        
        Returns:
            List of points along the Bezier curve
        """
        path = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.0
            path.append(control_points.evaluate(t))
        return path

    def composite_motions(self, motions: List[Dict[str, Any]], 
                         duration_frames: int) -> List[Point2D]:
        """
        Composite multiple motions together
        
        Args:
            motions: List of motion dictionaries with 'position', 'weight', etc.
            duration_frames: Total duration in frames
        
        Returns:
            List of composited positions
        """
        keyframes = [Point2D(0, 0) for _ in range(duration_frames)]

        for motion in motions:
            motion_frames = self.generate_keyframes(
                motion.get('start', Point2D(0, 0)),
                motion.get('end', Point2D(0, 0)),
                duration_frames,
                **motion.get('params', {})
            )
            weight = motion.get('weight', 1.0)

            for frame_idx, frame_pos in enumerate(motion_frames):
                keyframes[frame_idx] = keyframes[frame_idx] + (frame_pos * weight)

        return keyframes

    def clamp_to_canvas(self, point: Point2D) -> Point2D:
        """Clamp a point to the canvas boundaries"""
        x = max(0, min(self.width, point.x))
        y = max(0, min(self.height, point.y))
        return Point2D(x, y)

    def get_easing_function(self, easing_name: str) -> Callable[[float], float]:
        """
        Get an easing function by name
        
        Args:
            easing_name: Name of the easing function
        
        Returns:
            Easing function that maps [0, 1] to [0, 1]
        """
        easing_functions = {
            'linear': lambda t: t,
            'ease_in_quad': lambda t: t ** 2,
            'ease_out_quad': lambda t: 1 - (1 - t) ** 2,
            'ease_in_out_quad': lambda t: 2 * t ** 2 if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2,
            'ease_in_cubic': lambda t: t ** 3,
            'ease_out_cubic': lambda t: 1 - (1 - t) ** 3,
            'ease_in_out_cubic': lambda t: 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2,
            'ease_in_sine': lambda t: 1 - math.cos((t * math.pi) / 2),
            'ease_out_sine': lambda t: math.sin((t * math.pi) / 2),
            'ease_in_out_sine': lambda t: -(math.cos(math.pi * t) - 1) / 2,
        }
        return easing_functions.get(easing_name, easing_functions['linear'])


def create_complex_animation(renderer: AdvancedRenderer, 
                            width: int, height: int,
                            duration_frames: int) -> List[Point2D]:
    """
    Example function demonstrating complex animation creation
    
    Args:
        renderer: AdvancedRenderer instance
        width: Canvas width
        height: Canvas height
        duration_frames: Total duration in frames
    
    Returns:
        List of keyframes for complex motion
    """
    center = Point2D(width / 2, height / 2)
    
    # Composite multiple spiral motions with different parameters
    motions = [
        {
            'start': center,
            'end': center,
            'weight': 0.6,
            'params': {
                'center': center,
                'spiral_type': 'archimedean',
                'growth_rate': 0.3,
                'max_radius': 150
            }
        },
        {
            'start': center,
            'end': center,
            'weight': 0.4,
            'params': {
                'center': center,
                'spiral_type': 'logarithmic',
                'growth_rate': 0.05
            }
        }
    ]
    
    renderer.set_motion_type(MotionType.SPIRAL)
    return renderer.composite_motions(motions, duration_frames)


if __name__ == "__main__":
    # Example usage
    renderer = AdvancedRenderer(800, 600, fps=30)

    # Create a simple Bezier curve animation
    start = Point2D(100, 300)
    end = Point2D(700, 300)
    bezier = BezierControlPoints(
        p0=start,
        p1=Point2D(250, 100),
        p2=Point2D(550, 500),
        p3=end
    )

    renderer.set_motion_type(MotionType.BEZIER)
    renderer.add_bezier_curve(bezier)

    # Generate 30 frames (1 second at 30 fps)
    keyframes = renderer.generate_keyframes(start, end, 30)

    print(f"Generated {len(keyframes)} keyframes for Bezier curve animation")
    print(f"First keyframe: {keyframes[0].to_tuple()}")
    print(f"Last keyframe: {keyframes[-1].to_tuple()}")
