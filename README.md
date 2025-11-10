# Pixel Toy

A pixel art creator where Claude (the AI) can draw on a 128x128 canvas using function calling.

## Features

- **128x128 pixel canvas** with real-time drawing (displayed at 512x512px)
- **Chat interface** to interact with Claude
- **Cartesian coordinate system** - (0,0) is bottom-left, just like a graph!
- **Comprehensive drawing tools** Claude can use:
  - `set_pixel(x, y, r, g, b)` - Set individual pixels
  - `set_pixels([{x, y, r, g, b}, ...])` - Batch set multiple pixels efficiently
  - `draw_line(x1, y1, x2, y2, r, g, b)` - Draw straight lines
  - `draw_rect(x, y, width, height, r, g, b)` - Draw rectangle outlines
  - `fill_rect(x, y, width, height, r, g, b)` - Draw filled rectangles
  - `draw_circle(cx, cy, radius, r, g, b)` - Draw circle outlines
  - `fill_circle(cx, cy, radius, r, g, b)` - Draw filled circles
  - `draw_ellipse(cx, cy, rx, ry, r, g, b)` - Draw ellipse outlines
  - `fill_ellipse(cx, cy, rx, ry, r, g, b)` - Draw filled ellipses
  - `draw_bezier_quadratic(x0, y0, x1, y1, x2, y2, r, g, b)` - Draw smooth quadratic curves
  - `draw_bezier_cubic(x0, y0, x1, y1, x2, y2, x3, y3, r, g, b)` - Draw smooth cubic curves
  - `flood_fill(x, y, r, g, b)` - Paint bucket tool
  - `clear_canvas(r, g, b)` - Clear the canvas
  - `get_pixel(x, y)` - Read pixel colors

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```bash
   python3 main.py
   ```

## Usage

1. Type a message in the text box asking Claude to draw something
2. Press Enter or click "Send"
3. Watch Claude create pixel art on the canvas in real-time!

## Example Prompts

- "Draw a simple smiley face"
- "Create a small house with a red roof"
- "Draw a rainbow gradient"
- "Make a pixelated heart"
- "Draw a sunset scene"
- "Create a space invader"
- "Draw concentric circles in different colors"
- "Create a beach ball with colorful sections"
- "Draw a target with alternating red and white circles"
- "Make a simple landscape and use flood fill for the sky"
- "Draw a flower with curved petals using Bezier curves"
- "Create wavy ocean waves using curves"
- "Draw a smooth S-curve snake"

## Technical Details

- Built with Pygame and pygame-gui
- Uses Anthropic's Claude Sonnet 4 with function calling
- Canvas coordinates: (0,0) is bottom-left, (127,127) is top-right (Cartesian)
- RGB colors: 0-255 for each channel
- Drawing algorithms:
  - Bresenham's line algorithm for straight lines
  - Midpoint circle algorithm for circles
  - Midpoint ellipse algorithm for ellipses
  - Quadratic and cubic Bezier curves for smooth curves
  - Queue-based flood fill with 50k pixel safety limit
- Drawing happens in background thread to keep UI responsive
- Batch pixel operations for efficient complex drawings
