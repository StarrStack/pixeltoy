#!/usr/bin/env python3
"""
Pixel Toy - AI Pixel Art Creator with Pygame
A pixel art tool where Claude can draw via function calling.
"""

import pygame
import pygame_gui
import anthropic
import os
import json
import threading
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
CANVAS_SIZE = 128
PIXEL_SIZE = 4
CANVAS_PIXEL_WIDTH = CANVAS_SIZE * PIXEL_SIZE
CANVAS_PIXEL_HEIGHT = CANVAS_SIZE * PIXEL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)


class PixelCanvas:
    """Manages a pixel grid with drawing primitives."""

    def __init__(self, width=CANVAS_SIZE, height=CANVAS_SIZE):
        self.width = width
        self.height = height
        self.pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
        self.surface = pygame.Surface((width * PIXEL_SIZE, height * PIXEL_SIZE))
        self.redraw()

    def redraw(self):
        """Redraw the entire canvas."""
        for y in range(self.height):
            for x in range(self.width):
                color = self.pixels[y][x]
                rect = pygame.Rect(
                    x * PIXEL_SIZE,
                    y * PIXEL_SIZE,
                    PIXEL_SIZE,
                    PIXEL_SIZE
                )
                pygame.draw.rect(self.surface, color, rect)

    def set_pixel(self, x: int, y: int, r: int, g: int, b: int) -> bool:
        """Set a single pixel color. (0,0) is bottom-left."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            return False

        # Flip y-coordinate: 0 is at bottom, not top
        internal_y = (self.height - 1) - y
        self.pixels[internal_y][x] = (r, g, b)

        # Update just this pixel on the surface
        rect = pygame.Rect(x * PIXEL_SIZE, internal_y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)
        pygame.draw.rect(self.surface, (r, g, b), rect)
        return True

    def set_pixels(self, pixels: list) -> dict:
        """Set multiple pixels at once. Returns success count."""
        success_count = 0
        failed_count = 0

        for pixel in pixels:
            try:
                x = pixel['x']
                y = pixel['y']
                r = pixel['r']
                g = pixel['g']
                b = pixel['b']

                if self.set_pixel(x, y, r, g, b):
                    success_count += 1
                else:
                    failed_count += 1
            except (KeyError, TypeError):
                failed_count += 1

        return {
            "success": True,
            "pixels_set": success_count,
            "pixels_failed": failed_count,
            "total": len(pixels)
        }

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int) -> bool:
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        while True:
            self.set_pixel(x, y, r, g, b)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def draw_rect(self, x: int, y: int, width: int, height: int, r: int, g: int, b: int) -> bool:
        """Draw a rectangle outline."""
        self.draw_line(x, y, x + width - 1, y, r, g, b)
        self.draw_line(x, y + height - 1, x + width - 1, y + height - 1, r, g, b)
        self.draw_line(x, y, x, y + height - 1, r, g, b)
        self.draw_line(x + width - 1, y, x + width - 1, y + height - 1, r, g, b)
        return True

    def fill_rect(self, x: int, y: int, width: int, height: int, r: int, g: int, b: int) -> bool:
        """Draw a filled rectangle."""
        for py in range(y, min(y + height, self.height)):
            for px in range(x, min(x + width, self.width)):
                self.set_pixel(px, py, r, g, b)
        return True

    def clear(self, r: int = 255, g: int = 255, b: int = 255) -> bool:
        """Clear the canvas to a specific color."""
        self.fill_rect(0, 0, self.width, self.height, r, g, b)
        return True

    def draw_circle(self, cx: int, cy: int, radius: int, r: int, g: int, b: int) -> bool:
        """Draw a circle outline using midpoint circle algorithm."""
        x = radius
        y = 0
        err = 0

        while x >= y:
            # Draw 8 octants
            self.set_pixel(cx + x, cy + y, r, g, b)
            self.set_pixel(cx + y, cy + x, r, g, b)
            self.set_pixel(cx - y, cy + x, r, g, b)
            self.set_pixel(cx - x, cy + y, r, g, b)
            self.set_pixel(cx - x, cy - y, r, g, b)
            self.set_pixel(cx - y, cy - x, r, g, b)
            self.set_pixel(cx + y, cy - x, r, g, b)
            self.set_pixel(cx + x, cy - y, r, g, b)

            y += 1
            if err <= 0:
                err += 2 * y + 1
            else:
                x -= 1
                err += 2 * (y - x) + 1

        return True

    def fill_circle(self, cx: int, cy: int, radius: int, r: int, g: int, b: int) -> bool:
        """Draw a filled circle."""
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                if x * x + y * y <= radius * radius:
                    self.set_pixel(cx + x, cy + y, r, g, b)
        return True

    def draw_ellipse(self, cx: int, cy: int, rx: int, ry: int, r: int, g: int, b: int) -> bool:
        """Draw an ellipse outline."""
        # Using midpoint ellipse algorithm
        rx2 = rx * rx
        ry2 = ry * ry
        two_rx2 = 2 * rx2
        two_ry2 = 2 * ry2

        # Region 1
        x = 0
        y = ry
        px = 0
        py = two_rx2 * y

        self.set_pixel(cx + x, cy + y, r, g, b)
        self.set_pixel(cx - x, cy + y, r, g, b)
        self.set_pixel(cx + x, cy - y, r, g, b)
        self.set_pixel(cx - x, cy - y, r, g, b)

        p = round(ry2 - (rx2 * ry) + (0.25 * rx2))
        while px < py:
            x += 1
            px += two_ry2
            if p < 0:
                p += ry2 + px
            else:
                y -= 1
                py -= two_rx2
                p += ry2 + px - py

            self.set_pixel(cx + x, cy + y, r, g, b)
            self.set_pixel(cx - x, cy + y, r, g, b)
            self.set_pixel(cx + x, cy - y, r, g, b)
            self.set_pixel(cx - x, cy - y, r, g, b)

        # Region 2
        p = round(ry2 * (x + 0.5) * (x + 0.5) + rx2 * (y - 1) * (y - 1) - rx2 * ry2)
        while y > 0:
            y -= 1
            py -= two_rx2
            if p > 0:
                p += rx2 - py
            else:
                x += 1
                px += two_ry2
                p += rx2 - py + px

            self.set_pixel(cx + x, cy + y, r, g, b)
            self.set_pixel(cx - x, cy + y, r, g, b)
            self.set_pixel(cx + x, cy - y, r, g, b)
            self.set_pixel(cx - x, cy - y, r, g, b)

        return True

    def fill_ellipse(self, cx: int, cy: int, rx: int, ry: int, r: int, g: int, b: int) -> bool:
        """Draw a filled ellipse."""
        for y in range(-ry, ry + 1):
            for x in range(-rx, rx + 1):
                if (x * x) / (rx * rx) + (y * y) / (ry * ry) <= 1:
                    self.set_pixel(cx + x, cy + y, r, g, b)
        return True

    def draw_bezier_quadratic(self, x0: int, y0: int, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int, steps: int = 50) -> bool:
        """Draw a quadratic Bezier curve with 3 control points."""
        for i in range(steps + 1):
            t = i / steps
            # Quadratic Bezier formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
            x = int((1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2)
            y = int((1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t ** 2 * y2)
            self.set_pixel(x, y, r, g, b)
        return True

    def draw_bezier_cubic(self, x0: int, y0: int, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, r: int, g: int, b: int, steps: int = 50) -> bool:
        """Draw a cubic Bezier curve with 4 control points."""
        for i in range(steps + 1):
            t = i / steps
            # Cubic Bezier formula: B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3
            x = int((1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3 * x3)
            y = int((1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t ** 2 * y2 + t ** 3 * y3)
            self.set_pixel(x, y, r, g, b)
        return True

    def flood_fill(self, x: int, y: int, r: int, g: int, b: int) -> dict:
        """Flood fill starting from (x, y) with the specified color."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return {"success": False, "error": "Starting point out of bounds"}

        target_color = self.get_pixel(x, y)
        if target_color is None:
            return {"success": False, "error": "Invalid starting point"}

        fill_color = (r, g, b)
        if target_color == fill_color:
            return {"success": True, "pixels_filled": 0, "message": "Already that color"}

        # Use queue-based flood fill to avoid stack overflow
        pixels_filled = 0
        queue = [(x, y)]
        visited = set()

        while queue and pixels_filled < 50000:  # Safety limit
            cx, cy = queue.pop(0)

            if (cx, cy) in visited:
                continue
            if not (0 <= cx < self.width and 0 <= cy < self.height):
                continue

            current_color = self.get_pixel(cx, cy)
            if current_color != target_color:
                continue

            self.set_pixel(cx, cy, r, g, b)
            visited.add((cx, cy))
            pixels_filled += 1

            # Add neighbors
            queue.append((cx + 1, cy))
            queue.append((cx - 1, cy))
            queue.append((cx, cy + 1))
            queue.append((cx, cy - 1))

        return {
            "success": True,
            "pixels_filled": pixels_filled,
            "target_color": target_color,
            "fill_color": fill_color
        }

    def get_pixel(self, x: int, y: int) -> tuple:
        """Get the color of a pixel. (0,0) is bottom-left."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        # Flip y-coordinate: 0 is at bottom, not top
        internal_y = (self.height - 1) - y
        return self.pixels[internal_y][x]


class PixelToyApp:
    """Main application combining canvas and chat interface."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Pixel Toy - AI Pixel Art Creator")
        self.clock = pygame.time.Clock()
        self.running = True

        # Check for API key
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            print("ERROR: ANTHROPIC_API_KEY not found in .env file")
            print("Please create a .env file with your API key:")
            print("ANTHROPIC_API_KEY=your_api_key_here")
            self.running = False
            return

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.conversation_history = []

        # UI Manager
        self.ui_manager = pygame_gui.UIManager((WINDOW_WIDTH, WINDOW_HEIGHT))

        # Create canvas
        self.canvas = PixelCanvas()

        # Chat display (scrollable text box)
        self.chat_display = pygame_gui.elements.UITextBox(
            html_text="<font color='#00AA00'><b>System:</b> Welcome to Pixel Toy! Ask Claude to draw something.</font><br>"
                     "<font color='#00AA00'><b>System:</b> Canvas is 128x128 pixels. (0,0) is bottom-left.</font><br>",
            relative_rect=pygame.Rect(570, 20, 610, 550),
            manager=self.ui_manager
        )

        # Input box
        self.input_box = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(570, 590, 500, 40),
            manager=self.ui_manager
        )
        self.input_box.set_text("Draw a smiley face")

        # Send button
        self.send_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(1080, 590, 100, 40),
            text='Send',
            manager=self.ui_manager
        )

        # Tools for Claude
        self.tools = [
            {
                "name": "set_pixel",
                "description": "Set a single pixel to a specific RGB color. Coordinates are 0-indexed, with (0,0) at the bottom-left.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate (0-127, left to right)"},
                        "y": {"type": "integer", "description": "Y coordinate (0-127, bottom to top)"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["x", "y", "r", "g", "b"]
                }
            },
            {
                "name": "set_pixels",
                "description": "Set multiple pixels at once in a single batch operation. Much more efficient than calling set_pixel multiple times. Use this for drawing complex shapes or filling areas.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pixels": {
                            "type": "array",
                            "description": "Array of pixel data",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "integer", "description": "X coordinate (0-127)"},
                                    "y": {"type": "integer", "description": "Y coordinate (0-127)"},
                                    "r": {"type": "integer", "description": "Red value (0-255)"},
                                    "g": {"type": "integer", "description": "Green value (0-255)"},
                                    "b": {"type": "integer", "description": "Blue value (0-255)"}
                                },
                                "required": ["x", "y", "r", "g", "b"]
                            }
                        }
                    },
                    "required": ["pixels"]
                }
            },
            {
                "name": "draw_line",
                "description": "Draw a line from (x1,y1) to (x2,y2) with the specified RGB color.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "integer", "description": "Starting X coordinate"},
                        "y1": {"type": "integer", "description": "Starting Y coordinate"},
                        "x2": {"type": "integer", "description": "Ending X coordinate"},
                        "y2": {"type": "integer", "description": "Ending Y coordinate"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["x1", "y1", "x2", "y2", "r", "g", "b"]
                }
            },
            {
                "name": "draw_rect",
                "description": "Draw a rectangle outline.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Top-left X coordinate"},
                        "y": {"type": "integer", "description": "Top-left Y coordinate"},
                        "width": {"type": "integer", "description": "Width in pixels"},
                        "height": {"type": "integer", "description": "Height in pixels"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["x", "y", "width", "height", "r", "g", "b"]
                }
            },
            {
                "name": "fill_rect",
                "description": "Draw a filled rectangle.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Top-left X coordinate"},
                        "y": {"type": "integer", "description": "Top-left Y coordinate"},
                        "width": {"type": "integer", "description": "Width in pixels"},
                        "height": {"type": "integer", "description": "Height in pixels"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["x", "y", "width", "height", "r", "g", "b"]
                }
            },
            {
                "name": "clear_canvas",
                "description": "Clear the entire canvas to a specific color (default white).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "r": {"type": "integer", "description": "Red value (0-255)", "default": 255},
                        "g": {"type": "integer", "description": "Green value (0-255)", "default": 255},
                        "b": {"type": "integer", "description": "Blue value (0-255)", "default": 255}
                    }
                }
            },
            {
                "name": "get_pixel",
                "description": "Get the current RGB color of a specific pixel.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"}
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "draw_circle",
                "description": "Draw a circle outline with center at (cx, cy) and given radius.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cx": {"type": "integer", "description": "Center X coordinate"},
                        "cy": {"type": "integer", "description": "Center Y coordinate"},
                        "radius": {"type": "integer", "description": "Radius in pixels"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["cx", "cy", "radius", "r", "g", "b"]
                }
            },
            {
                "name": "fill_circle",
                "description": "Draw a filled circle with center at (cx, cy) and given radius.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cx": {"type": "integer", "description": "Center X coordinate"},
                        "cy": {"type": "integer", "description": "Center Y coordinate"},
                        "radius": {"type": "integer", "description": "Radius in pixels"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["cx", "cy", "radius", "r", "g", "b"]
                }
            },
            {
                "name": "draw_ellipse",
                "description": "Draw an ellipse outline with center at (cx, cy) and given radii.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cx": {"type": "integer", "description": "Center X coordinate"},
                        "cy": {"type": "integer", "description": "Center Y coordinate"},
                        "rx": {"type": "integer", "description": "X radius in pixels"},
                        "ry": {"type": "integer", "description": "Y radius in pixels"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["cx", "cy", "rx", "ry", "r", "g", "b"]
                }
            },
            {
                "name": "fill_ellipse",
                "description": "Draw a filled ellipse with center at (cx, cy) and given radii.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cx": {"type": "integer", "description": "Center X coordinate"},
                        "cy": {"type": "integer", "description": "Center Y coordinate"},
                        "rx": {"type": "integer", "description": "X radius in pixels"},
                        "ry": {"type": "integer", "description": "Y radius in pixels"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["cx", "cy", "rx", "ry", "r", "g", "b"]
                }
            },
            {
                "name": "flood_fill",
                "description": "Fill a contiguous area with a color, starting from point (x, y). Similar to paint bucket tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Starting X coordinate"},
                        "y": {"type": "integer", "description": "Starting Y coordinate"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["x", "y", "r", "g", "b"]
                }
            },
            {
                "name": "draw_bezier_quadratic",
                "description": "Draw a smooth quadratic Bezier curve using 3 control points. Great for simple curves.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x0": {"type": "integer", "description": "Start point X"},
                        "y0": {"type": "integer", "description": "Start point Y"},
                        "x1": {"type": "integer", "description": "Control point X"},
                        "y1": {"type": "integer", "description": "Control point Y"},
                        "x2": {"type": "integer", "description": "End point X"},
                        "y2": {"type": "integer", "description": "End point Y"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["x0", "y0", "x1", "y1", "x2", "y2", "r", "g", "b"]
                }
            },
            {
                "name": "draw_bezier_cubic",
                "description": "Draw a smooth cubic Bezier curve using 4 control points. Allows more complex curves with two control points.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x0": {"type": "integer", "description": "Start point X"},
                        "y0": {"type": "integer", "description": "Start point Y"},
                        "x1": {"type": "integer", "description": "First control point X"},
                        "y1": {"type": "integer", "description": "First control point Y"},
                        "x2": {"type": "integer", "description": "Second control point X"},
                        "y2": {"type": "integer", "description": "Second control point Y"},
                        "x3": {"type": "integer", "description": "End point X"},
                        "y3": {"type": "integer", "description": "End point Y"},
                        "r": {"type": "integer", "description": "Red value (0-255)"},
                        "g": {"type": "integer", "description": "Green value (0-255)"},
                        "b": {"type": "integer", "description": "Blue value (0-255)"}
                    },
                    "required": ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b"]
                }
            }
        ]

        self.processing = False

    def add_to_chat(self, sender: str, message: str, color: str = "#FFFFFF"):
        """Add a message to the chat display."""
        current_html = self.chat_display.html_text
        new_message = f"<font color='{color}'><b>{sender}:</b> {message}</font><br>"
        self.chat_display.html_text = current_html + new_message
        self.chat_display.rebuild()
        # Scroll to bottom (if scroll bar exists)
        if self.chat_display.scroll_bar is not None:
            self.chat_display.scroll_bar.scroll_position = self.chat_display.scroll_bar.bottom_limit

    def send_message(self):
        """Send user message to Claude."""
        if self.processing:
            return

        message = self.input_box.get_text().strip()
        if not message:
            return

        self.input_box.set_text("")
        self.add_to_chat("You", message, "#3366FF")

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Process in background thread
        self.processing = True
        thread = threading.Thread(target=self.process_with_claude, daemon=True)
        thread.start()

    def execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a drawing tool."""
        try:
            if tool_name == "set_pixel":
                success = self.canvas.set_pixel(
                    tool_input['x'], tool_input['y'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "set_pixels":
                return self.canvas.set_pixels(tool_input['pixels'])

            elif tool_name == "draw_line":
                success = self.canvas.draw_line(
                    tool_input['x1'], tool_input['y1'],
                    tool_input['x2'], tool_input['y2'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "draw_rect":
                success = self.canvas.draw_rect(
                    tool_input['x'], tool_input['y'],
                    tool_input['width'], tool_input['height'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "fill_rect":
                success = self.canvas.fill_rect(
                    tool_input['x'], tool_input['y'],
                    tool_input['width'], tool_input['height'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "clear_canvas":
                r = tool_input.get('r', 255)
                g = tool_input.get('g', 255)
                b = tool_input.get('b', 255)
                success = self.canvas.clear(r, g, b)
                return {"success": success}

            elif tool_name == "get_pixel":
                color = self.canvas.get_pixel(tool_input['x'], tool_input['y'])
                if color:
                    return {"r": color[0], "g": color[1], "b": color[2]}
                return {"error": "Invalid coordinates"}

            elif tool_name == "draw_circle":
                success = self.canvas.draw_circle(
                    tool_input['cx'], tool_input['cy'], tool_input['radius'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "fill_circle":
                success = self.canvas.fill_circle(
                    tool_input['cx'], tool_input['cy'], tool_input['radius'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "draw_ellipse":
                success = self.canvas.draw_ellipse(
                    tool_input['cx'], tool_input['cy'],
                    tool_input['rx'], tool_input['ry'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "fill_ellipse":
                success = self.canvas.fill_ellipse(
                    tool_input['cx'], tool_input['cy'],
                    tool_input['rx'], tool_input['ry'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "flood_fill":
                return self.canvas.flood_fill(
                    tool_input['x'], tool_input['y'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )

            elif tool_name == "draw_bezier_quadratic":
                success = self.canvas.draw_bezier_quadratic(
                    tool_input['x0'], tool_input['y0'],
                    tool_input['x1'], tool_input['y1'],
                    tool_input['x2'], tool_input['y2'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            elif tool_name == "draw_bezier_cubic":
                success = self.canvas.draw_bezier_cubic(
                    tool_input['x0'], tool_input['y0'],
                    tool_input['x1'], tool_input['y1'],
                    tool_input['x2'], tool_input['y2'],
                    tool_input['x3'], tool_input['y3'],
                    tool_input['r'], tool_input['g'], tool_input['b']
                )
                return {"success": success}

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"error": str(e)}

    def process_with_claude(self):
        """Send conversation to Claude and process tool calls."""
        try:
            system_message = """You are a pixel art assistant. You can draw on a 128x128 pixel canvas using the provided tools.
The canvas uses standard Cartesian coordinates: (0,0) is at the bottom-left corner and (127,127) is at the top-right.
X coordinates go from 0 (left) to 127 (right). Y coordinates go from 0 (bottom) to 127 (top).

IMPORTANT: When drawing complex shapes or many pixels, use the set_pixels tool to set multiple pixels in a single batch operation.
This is much more efficient than calling set_pixel repeatedly. You can set hundreds or even thousands of pixels in one call.

Be concise in your responses and then use the tools to create the art."""

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_message,
                messages=self.conversation_history,
                tools=self.tools
            )

            # Process response
            assistant_message = {"role": "assistant", "content": []}
            tool_results = []

            for block in response.content:
                if block.type == "text":
                    assistant_message["content"].append(block)
                    # Schedule UI update in main thread
                    pygame.event.post(pygame.event.Event(
                        pygame.USEREVENT,
                        {"action": "add_chat", "sender": "Claude", "message": block.text, "color": "#00AA00"}
                    ))

                elif block.type == "tool_use":
                    assistant_message["content"].append(block)
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

                    # Show tool execution
                    tool_desc = f"{block.name}({block.input})"
                    pygame.event.post(pygame.event.Event(
                        pygame.USEREVENT,
                        {"action": "add_chat", "sender": "System", "message": f"Executed: {tool_desc}", "color": "#888888"}
                    ))

            # Add assistant message to history
            self.conversation_history.append(assistant_message)

            # If there were tool calls, send results back
            if tool_results:
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
                # Recursively process to get final response
                self.process_with_claude()
            else:
                self.processing = False

        except Exception as e:
            pygame.event.post(pygame.event.Event(
                pygame.USEREVENT,
                {"action": "add_chat", "sender": "Error", "message": str(e), "color": "#FF0000"}
            ))
            self.processing = False

    def handle_events(self):
        """Handle pygame events."""
        time_delta = self.clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.USEREVENT:
                if event.__dict__.get('action') == 'add_chat':
                    self.add_to_chat(event.sender, event.message, event.color)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.send_button:
                    self.send_message()

            if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                if event.ui_element == self.input_box:
                    self.send_message()

            self.ui_manager.process_events(event)

        self.ui_manager.update(time_delta)

    def render(self):
        """Render the application."""
        self.screen.fill(DARK_GRAY)

        # Draw canvas with border
        canvas_x, canvas_y = 20, 20
        border_rect = pygame.Rect(canvas_x - 2, canvas_y - 2, CANVAS_PIXEL_WIDTH + 4, CANVAS_PIXEL_HEIGHT + 4)
        pygame.draw.rect(self.screen, GRAY, border_rect)

        # Draw canvas
        self.screen.blit(self.canvas.surface, (canvas_x, canvas_y))

        # Draw canvas label
        font = pygame.font.Font(None, 24)
        label = font.render("Canvas (128x128)", True, WHITE)
        self.screen.blit(label, (canvas_x, canvas_y + CANVAS_PIXEL_HEIGHT + 10))

        # Draw UI
        self.ui_manager.draw_ui(self.screen)

        pygame.display.flip()

    def run(self):
        """Main application loop."""
        if not self.running:
            return

        while self.running:
            self.handle_events()
            self.render()

        pygame.quit()


def main():
    app = PixelToyApp()
    app.run()


if __name__ == "__main__":
    main()
