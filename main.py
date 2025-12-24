"""
HTML Motion to GIF Converter
Converts HTML motion codes into animated GIFs with customizable options
"""

import os
import tempfile
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw
import io
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max


class MotionType(Enum):
    """Enumeration of motion types"""
    TRANSFORM = "transform"
    ANIMATION = "animation"
    TRANSITION = "transition"
    KEYFRAMES = "keyframes"
    CUSTOM_JS = "custom_js"


@dataclass
class GIFConfig:
    """Configuration for GIF generation"""
    max_pixels: int = 3333
    custom_pixels: Optional[int] = None
    bg_color: str = "#FFFFFF"
    transparent: bool = False
    output_pixels: Optional[int] = None
    frame_duration: int = 50  # milliseconds
    loop: int = 0  # 0 = infinite


class MotionDetector:
    """Detects and extracts different motion types from HTML"""
    
    @staticmethod
    def detect_motions(html_code: str) -> List[Tuple[MotionType, str, dict]]:
        """
        Detect all motion types in HTML code
        Returns: List of (motion_type, motion_code, metadata)
        """
        motions = []
        
        # Detect CSS animations
        animation_pattern = r'@keyframes\s+(\w+)\s*\{([^}]+(?: \{[^}]*\}[^}]*)*)\}'
        for match in re.finditer(animation_pattern, html_code):
            animation_name = match.group(1)
            animation_code = match.group(2)
            motions.append((
                MotionType.KEYFRAMES,
                animation_code,
                {"name": animation_name, "type": "css_keyframes"}
            ))
        
        # Detect CSS transforms
        transform_pattern = r'transform:\s*([^;]+);'
        for match in re.finditer(transform_pattern, html_code):
            transform_code = match.group(1)
            motions.append((
                MotionType.TRANSFORM,
                transform_code,
                {"type": "css_transform"}
            ))
        
        # Detect CSS transitions
        transition_pattern = r'transition:\s*([^;]+);'
        for match in re.finditer(transition_pattern, html_code):
            transition_code = match.group(1)
            motions.append((
                MotionType.TRANSITION,
                transition_code,
                {"type": "css_transition"}
            ))
        
        # Detect CSS animations
        anim_pattern = r'animation:\s*([^;]+);'
        for match in re.finditer(anim_pattern, html_code):
            anim_code = match.group(1)
            motions.append((
                MotionType.ANIMATION,
                anim_code,
                {"type": "css_animation"}
            ))
        
        # Detect JavaScript animations
        js_pattern = r'(?: requestAnimationFrame|setInterval|setTimeout)\s*\(\s*function[^}]*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}'
        for match in re.finditer(js_pattern, html_code):
            js_code = match.group(1)
            motions.append((
                MotionType.CUSTOM_JS,
                js_code,
                {"type": "javascript_animation"}
            ))
        
        return motions


class HTMLMotionParser:
    """Parses HTML and extracts SVG/Canvas elements and their styles"""
    
    @staticmethod
    def extract_elements(html_code: str) -> List[dict]:
        """Extract visual elements (SVG, Canvas, DOM) from HTML"""
        elements = []
        
        # Extract SVG elements
        svg_pattern = r'<svg[^>]*>(.*?)</svg>'
        for match in re.finditer(svg_pattern, html_code, re.DOTALL):
            elements.append({
                "type": "svg",
                "content": match.group(0),
                "raw": match.group(1)
            })
        
        # Extract style definitions
        style_pattern = r'<style[^>]*>(.*?)</style>'
        for match in re.finditer(style_pattern, html_code, re.DOTALL):
            elements.append({
                "type": "style",
                "content": match.group(1)
            })
        
        # Extract DIVs with styles/classes
        div_pattern = r'<div[^>]*(class|style)[^>]*>(.*?)</div>'
        for match in re.finditer(div_pattern, html_code, re.DOTALL):
            elements.append({
                "type": "div",
                "content": match.group(0),
                "attr": match.group(1)
            })
        
        return elements


class GIFGenerator:
    """Generates GIFs from motion descriptions"""
    
    def __init__(self, config: GIFConfig):
        self.config = config
        self.width = config.output_pixels or config.custom_pixels or config.max_pixels
        self.height = self.width
    
    def generate_frames(self, motion_type: MotionType, motion_data: str, 
                       frame_count: int = 30) -> List[Image.Image]:
        """Generate PIL Image frames based on motion type"""
        frames = []
        
        try:
            if motion_type == MotionType.KEYFRAMES:
                frames = self._generate_keyframe_animation(motion_data, frame_count)
            elif motion_type == MotionType.TRANSFORM:
                frames = self._generate_transform_animation(motion_data, frame_count)
            elif motion_type == MotionType.TRANSITION:
                frames = self._generate_transition_animation(motion_data, frame_count)
            elif motion_type == MotionType.ANIMATION:
                frames = self._generate_css_animation(motion_data, frame_count)
            elif motion_type == MotionType.CUSTOM_JS:
                frames = self._generate_js_animation(motion_data, frame_count)
        except Exception as e:
            logger.error(f"Error generating frames for {motion_type}: {e}")
            frames = [self._create_blank_frame()]
        
        return frames if frames else [self._create_blank_frame()]
    
    def _create_blank_frame(self) -> Image.Image:
        """Create a blank frame"""
        if self.config.transparent:
            return Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        else:
            bg_color = self._hex_to_rgb(self.config.bg_color)
            return Image.new('RGB', (self.width, self.height), bg_color)
    
    def _generate_keyframe_animation(self, keyframe_data: str, frames: int) -> List[Image.Image]:
        """Generate frames from keyframe animation"""
        frames_list = []
        keyframe_values = self._parse_keyframes(keyframe_data)
        
        for i in range(frames):
            progress = i / frames
            frame = self._create_blank_frame()
            draw = ImageDraw.Draw(frame)
            
            # Interpolate between keyframes
            current_state = self._interpolate_keyframes(keyframe_values, progress)
            self._draw_motion(draw, current_state, i, frames)
            frames_list.append(frame)
        
        return frames_list
    
    def _generate_transform_animation(self, transform_data: str, frame_count: int) -> List[Image.Image]:
        """Generate frames from CSS transform"""
        frames_list = []
        
        for i in range(frame_count):
            progress = i / frame_count
            frame = self._create_blank_frame()
            draw = ImageDraw.Draw(frame)
            
            # Parse and apply transform
            transform_state = self._apply_transform(transform_data, progress)
            self._draw_motion(draw, transform_state, i, frame_count)
            frames_list.append(frame)
        
        return frames_list
    
    def _generate_transition_animation(self, transition_data: str, frame_count: int) -> List[Image.Image]:
        """Generate frames from CSS transition"""
        frames_list = []
        
        for i in range(frame_count):
            progress = i / frame_count
            frame = self._create_blank_frame()
            draw = ImageDraw.Draw(frame)
            
            # Parse and apply transition
            transition_state = self._apply_transition(transition_data, progress)
            self._draw_motion(draw, transition_state, i, frame_count)
            frames_list.append(frame)
        
        return frames_list
    
    def _generate_css_animation(self, animation_data: str, frame_count: int) -> List[Image.Image]:
        """Generate frames from CSS animation property"""
        frames_list = []
        
        for i in range(frame_count):
            progress = i / frame_count
            frame = self._create_blank_frame()
            draw = ImageDraw.Draw(frame)
            
            # Parse animation properties
            anim_state = self._apply_animation(animation_data, progress)
            self._draw_motion(draw, anim_state, i, frame_count)
            frames_list.append(frame)
        
        return frames_list
    
    def _generate_js_animation(self, js_code: str, frame_count: int) -> List[Image.Image]:
        """Generate frames from JavaScript animation code"""
        frames_list = []
        
        for i in range(frame_count):
            progress = i / frame_count
            frame = self._create_blank_frame()
            draw = ImageDraw.Draw(frame)
            
            # Extract numerical values from JS code and create animation
            values = self._extract_js_values(js_code)
            js_state = self._apply_js_animation(values, progress)
            self._draw_motion(draw, js_state, i, frame_count)
            frames_list.append(frame)
        
        return frames_list
    
    def _parse_keyframes(self, keyframe_data: str) -> dict:
        """Parse keyframe percentages and properties"""
        keyframes = {}
        # Parse 0%, 50%, 100% format
        pattern = r'(\d+)%?\s*\{([^}]+)\}'
        for match in re.finditer(pattern, keyframe_data):
            percentage = int(match.group(1))
            properties = match.group(2)
            keyframes[percentage] = self._parse_css_properties(properties)
        return keyframes
    
    def _interpolate_keyframes(self, keyframes: dict, progress: float) -> dict:
        """Interpolate between keyframes based on progress"""
        if not keyframes:
            return {}
        
        progress_percent = progress * 100
        sorted_keys = sorted(keyframes.keys())
        
        # Find surrounding keyframes
        prev_key = sorted_keys[0]
        next_key = sorted_keys[-1]
        
        for key in sorted_keys:
            if key <= progress_percent:
                prev_key = key
            if key >= progress_percent and key > prev_key:
                next_key = key
                break
        
        if prev_key == next_key:
            return keyframes[prev_key]
        
        # Linear interpolation
        range_percent = next_key - prev_key
        local_progress = (progress_percent - prev_key) / range_percent if range_percent > 0 else 0
        
        prev_state = keyframes[prev_key]
        next_state = keyframes[next_key]
        
        return self._interpolate_states(prev_state, next_state, local_progress)
    
    def _apply_transform(self, transform_data: str, progress: float) -> dict:
        """Apply CSS transform with animation progress"""
        state = {
            "rotate": 0,
            "scale": 1,
            "translateX": 0,
            "translateY": 0,
            "skew": 0
        }
        
        # Parse transform functions
        rotate_match = re.search(r'rotate\((\d+(?:\.\d+)?)(deg|turn|rad)\)', transform_data)
        if rotate_match:
            value = float(rotate_match.group(1))
            unit = rotate_match.group(2)
            state["rotate"] = self._convert_to_degrees(value, unit) * progress
        
        scale_match = re.search(r'scale\((\d+(?:\.\d+)?)\)', transform_data)
        if scale_match:
            value = float(scale_match.group(1))
            state["scale"] = 1 + (value - 1) * progress
        
        translate_match = re.search(r'translateX\((\d+(?:\.\d+)?)px\)', transform_data)
        if translate_match:
            value = float(translate_match.group(1))
            state["translateX"] = value * progress
        
        translate_y_match = re.search(r'translateY\((\d+(?:\.\d+)?)px\)', transform_data)
        if translate_y_match:
            value = float(translate_y_match.group(1))
            state["translateY"] = value * progress
        
        return state
    
    def _apply_transition(self, transition_data: str, progress: float) -> dict:
        """Apply CSS transition animation"""
        return self._apply_transform(transition_data, progress)
    
    def _apply_animation(self, animation_data: str, progress: float) -> dict:
        """Apply CSS animation property"""
        return self._apply_transform(animation_data, progress)
    
    def _apply_js_animation(self, values: List[float], progress: float) -> dict:
        """Apply JavaScript animation state"""
        if not values:
            return {"x": 0, "y": 0, "opacity": 1}
        
        return {
            "x": (values[0] if len(values) > 0 else 0) * progress,
            "y": (values[1] if len(values) > 1 else 0) * progress,
            "opacity": 1 - (values[2] if len(values) > 2 else 0) * progress
        }
    
    def _extract_js_values(self, js_code: str) -> List[float]:
        """Extract numerical values from JavaScript code"""
        numbers = re.findall(r'(\d+(?:\.\d+)?)', js_code)
        return [float(n) for n in numbers[:3]]  # Get first 3 numbers
    
    def _parse_css_properties(self, properties: str) -> dict:
        """Parse CSS properties from a string"""
        state = {}
        pairs = re.findall(r'(\w+)\s*:\s*([^;]+)', properties)
        for key, value in pairs:
            state[key] = value.strip()
        return state
    
    def _interpolate_states(self, prev_state: dict, next_state: dict, progress: float) -> dict:
        """Interpolate between two animation states"""
        result = {}
        for key in set(prev_state.keys()) | set(next_state.keys()):
            prev_val = prev_state.get(key, 0)
            next_val = next_state.get(key, 0)
            
            # Try to extract numbers
            try:
                prev_num = float(re.findall(r'\d+(?:\.\d+)?', str(prev_val))[0]) if re.findall(r'\d+(?:\.\d+)?', str(prev_val)) else 0
                next_num = float(re.findall(r'\d+(?:\.\d+)?', str(next_val))[0]) if re.findall(r'\d+(?:\.\d+)?', str(next_val)) else 0
                result[key] = prev_num + (next_num - prev_num) * progress
            except:
                result[key] = next_val
        
        return result
    
    def _draw_motion(self, draw: ImageDraw.ImageDraw, state: dict, frame_num: int, total_frames: int):
        """Draw motion state on frame"""
        # Draw center circle that moves based on animation state
        center_x = self.width // 2
        center_y = self.height // 2
        radius = 50
        
        # Apply transform state
        if "rotate" in state:
            pass
        
        if "translateX" in state:
            center_x += int(state["translateX"])
        
        if "translateY" in state:
            center_y += int(state["translateY"])
        
        if "scale" in state:
            radius = int(radius * state["scale"])
        
        # Draw circle
        bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        
        if self.config.transparent:
            alpha = int(255 * state.get("opacity", 1))
            color = (66, 135, 245, alpha)
        else:
            color = (66, 135, 245)
        
        draw.ellipse(bbox, fill=color)
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _convert_to_degrees(self, value: float, unit: str) -> float:
        """Convert angle to degrees"""
        if unit == "deg":
            return value
        elif unit == "turn":
            return value * 360
        elif unit == "rad":
            return value * (180 / 3.14159)
        return value
    
    def save_gif(self, frames: List[Image.Image], filename: str = "output.gif"):
        """Save frames as animated GIF"""
        if not frames:
            frames = [self._create_blank_frame()]
        
        # Ensure all frames have same mode
        if self.config.transparent:
            frames = [frame.convert("RGBA") for frame in frames]
        else:
            frames = [frame.convert("RGB") for frame in frames]
        
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:] if len(frames) > 1 else [],
            duration=self.config.frame_duration,
            loop=self.config.loop,
            optimize=False
        )
        
        return filename
    
    def get_gif_bytes(self, frames: List[Image.Image]) -> bytes:
        """Get GIF as bytes without saving to file"""
        if not frames:
            frames = [self._create_blank_frame()]
        
        # Ensure all frames have same mode
        if self.config.transparent:
            frames = [frame.convert("RGBA") for frame in frames]
        else:
            frames = [frame.convert("RGB") for frame in frames]
        
        gif_buffer = io.BytesIO()
        frames[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:] if len(frames) > 1 else [],
            duration=self.config.frame_duration,
            loop=self.config.loop,
            optimize=False
        )
        gif_buffer.seek(0)
        return gif_buffer.getvalue()


@app.route('/convert', methods=['POST'])
def convert_html_to_gif():
    """
    Convert HTML motion code to GIF
    
    Request JSON:
    {
        "html_code": "<your html here>",
        "transparent": false,
        "bg_color": "#FFFFFF",
        "custom_pixels": 2000,
        "frame_count": 30
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'html_code' not in data:
            return jsonify({"error": "html_code is required"}), 400
        
        html_code = data['html_code']
        config = GIFConfig(
            transparent=data.get('transparent', False),
            bg_color=data.get('bg_color', '#FFFFFF'),
            custom_pixels=data.get('custom_pixels', None),
            frame_duration=data.get('frame_duration', 50),
        )
        
        # Detect motions
        detector = MotionDetector()
        motions = detector.detect_motions(html_code)
        
        if not motions:
            logger.warning("No motions detected in HTML code")
            motions = [(MotionType.TRANSFORM, "rotate(360deg)", {})]
        
        # Generate GIFs for each motion
        generator = GIFGenerator(config)
        frame_count = data.get('frame_count', 30)
        
        all_gifs = []
        for idx, (motion_type, motion_data, metadata) in enumerate(motions):
            frames = generator.generate_frames(motion_type, motion_data, frame_count)
            gif_bytes = generator.get_gif_bytes(frames)
            all_gifs.append({
                "motion_index": idx,
                "motion_type": motion_type.value,
                "metadata": metadata,
                "gif_data": gif_bytes.hex()
            })
        
        return jsonify({
            "success": True,
            "motions_detected": len(motions),
            "gifs": all_gifs,
            "config": {
                "output_pixels": config.custom_pixels or config.max_pixels,
                "transparent": config.transparent,
                "bg_color": config.bg_color
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error converting HTML to GIF: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/convert/single', methods=['POST'])
def convert_single_gif():
    """
    Convert HTML motion code to a single merged GIF
    Returns the GIF file directly
    """
    try:
        # Check if JSON or form data
        if request.is_json:
            data = request.get_json()
            html_code = data.get('html_code')
        else:
            html_code = request.form.get('html_code')
        
        if not html_code:
            return jsonify({"error": "html_code is required"}), 400
        
        # Parse config
        transparent = request.form.get('transparent', 'false').lower() == 'true' if not request.is_json else request.json.get('transparent', False)
        bg_color = request.form.get('bg_color', '#FFFFFF') if not request.is_json else request.json.get('bg_color', '#FFFFFF')
        custom_pixels = request.form.get('custom_pixels') if not request.is_json else request.json.get('custom_pixels')
        frame_count = int(request.form.get('frame_count', 30) if not request.is_json else request.json.get('frame_count', 30))
        
        if custom_pixels:
            custom_pixels = int(custom_pixels)
        
        config = GIFConfig(
            transparent=transparent,
            bg_color=bg_color,
            custom_pixels=custom_pixels,
        )
        
        # Detect motions
        detector = MotionDetector()
        motions = detector.detect_motions(html_code)
        
        if not motions:
            logger.warning("No motions detected in HTML code")
            motions = [(MotionType.TRANSFORM, "rotate(360deg)", {})]
        
        # Generate combined frames
        generator = GIFGenerator(config)
        all_frames = []
        
        for motion_type, motion_data, metadata in motions:
            frames = generator.generate_frames(motion_type, motion_data, frame_count)
            all_frames.extend(frames)
        
        if not all_frames:
            all_frames = [generator._create_blank_frame()]
        
        gif_bytes = generator.get_gif_bytes(all_frames)
        
        return send_file(
            io.BytesIO(gif_bytes),
            mimetype='image/gif',
            as_attachment=True,
            download_name='animation.gif'
        )
    
    except Exception as e:
        logger.error(f"Error converting to single GIF: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200


@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        "name": "HTML Motion to GIF Converter",
        "version": "1.0.0",
        "endpoints": {
            "POST /convert": "Convert HTML to multiple GIFs (one per motion)",
            "POST /convert/single": "Convert HTML to single merged GIF",
            "GET /health": "Health check"
        },
        "parameters": {
            "html_code": "HTML code containing motion (required)",
            "transparent": "Generate transparent GIF (boolean, default: false)",
            "bg_color": "Background color hex code (default: #FFFFFF)",
            "custom_pixels": "Custom output size in pixels (default: 3333 max)",
            "frame_count": "Number of frames (default: 30)"
        },
        "example": {
            "html_code": "<style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style><div style='animation: spin 2s infinite;'></div>",
            "transparent": False,
            "bg_color": "#FFFFFF",
            "custom_pixels": 1920,
            "frame_count": 60
        }
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
