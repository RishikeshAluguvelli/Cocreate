"""Tool definitions handed to the planner."""
from __future__ import annotations

PHOTO_IDS = {"type": "array", "items": {"type": "string"},
             "description": "Photo ids to edit. Use [\"all\"] for every photo in the shoot."}
SOURCE = {"type": "string", "enum": ["head", "original", "bracket", "reference"],
          "description": "head = current edited version; original = finished photo before any chat edits; bracket = an original exposure (index, default middle); reference = customer-supplied reference (index)."}

TOOLS: list[dict] = [
    {
        "name": "inspect_photo",
        "description": "Look closer at a photo (or a bracket / reference) at higher resolution, optionally cropped to a box. Returns the image so you can check a colour, a detail, or whether an object is present.",
        "input_schema": {"type": "object", "properties": {
            "photo_id": {"type": "string"},
            "source": SOURCE,
            "index": {"type": "integer", "description": "bracket/reference index (0-based)"},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                    "description": "normalized crop [x0,y0,x1,y1]"},
        }, "required": ["photo_id"]},
    },
    {
        "name": "sample_color",
        "description": "Segment a surface by name (e.g. 'wall', 'left accent wall', 'floor tiles') on the chosen source image and return its paint colour as a hex code (median of mid-lit pixels). Use source='bracket' to read the true colour from an original exposure, 'reference' to read it from a customer reference image.",
        "input_schema": {"type": "object", "properties": {
            "photo_id": {"type": "string"},
            "target": {"type": "string", "description": "what to sample, short noun phrase"},
            "source": SOURCE,
            "index": {"type": "integer"},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                    "description": "optional normalized box to restrict the search"},
            "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "description": "optional normalized points the user placed on the surface: the colour is read from the surface grown around them (most precise)"},
        }, "required": ["photo_id", "target"]},
    },
    {
        "name": "repaint",
        "description": "Repaint a surface (walls, floor, cabinets, door) to a hex colour while keeping its texture, shading and shadows. Segments `target` on each photo's head version; photos where the target is not found are skipped.",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "target": {"type": "string", "description": "surface to repaint, e.g. 'wall', 'grey accent wall', 'floor'"},
            "hex": {"type": "string", "description": "target colour like #3A5F6B"},
            "luminance_follow": {"type": "number", "description": "0..1, how much the brightness moves toward the target colour's brightness (default 0.7; use 0.1-0.2 for water, sky, lawn tints)"},
            "strength": {"type": "number", "description": "0..1 how far the hue/saturation moves toward the target (default 1 = full repaint; 0.4-0.6 = 'a bit more blue/green')"},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                    "description": "optional normalized box restricting which surface instance to repaint"},
            "painted_area": {"type": "boolean", "description": "include the region the user painted on this photo (see painted_area_by_user in the state); single photo only"},
            "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "description": "optional normalized points the user placed on the surface (points_placed_by_user in the state): the surface is grown from exactly those spots instead of detected; single photo only"},
        }, "required": ["photo_ids", "target", "hex"]},
    },
    {
        "name": "adjust",
        "description": "Slider adjustments on whole photos or a named region. All values -100..100, 0 = unchanged. brightness is exposure (+100 = +1.5 EV).",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "brightness": {"type": "number"}, "contrast": {"type": "number"}, "saturation": {"type": "number"},
            "temperature": {"type": "number", "description": "+ warmer, - cooler"},
            "shadows": {"type": "number", "description": "+ lifts dark areas"},
            "highlights": {"type": "number", "description": "+ recovers bright areas"},
            "sharpen": {"type": "number", "description": "0..100 unsharp amount"},
            "region": {"type": "string", "description": "optional: only adjust this thing, e.g. 'ceiling', 'window', 'dark corner on the left'"},
        }, "required": ["photo_ids"]},
    },
    {
        "name": "draw_lot_lines",
        "description": "Overlay property / lot lines on an aerial photo as a polygon of normalized points.",
        "input_schema": {"type": "object", "properties": {
            "photo_id": {"type": "string"},
            "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "minItems": 2, "description": "[[x,y],...] normalized 0..1, in order around the parcel"},
            "color": {"type": "string", "description": "hex, default #FFD60A (yellow)"},
            "style": {"type": "string", "enum": ["solid", "dashed", "dotted"]},
            "fill_opacity": {"type": "number", "description": "0..1 translucent fill, default 0.10"},
            "label": {"type": "string", "description": "optional text at the centre, e.g. 'LOT 12 · 0.42 ac'"},
            "closed": {"type": "boolean", "description": "default true"},
        }, "required": ["photo_id", "points"]},
    },
    {
        "name": "add_watermark",
        "description": "Stamp a text or logo watermark on photos (same placement on every photo).",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "text": {"type": "string", "description": "watermark text; ignored when use_logo is true"},
            "use_logo": {"type": "boolean", "description": "use the logo the user uploaded (only if the context says a logo exists)"},
            "position": {"type": "string", "enum": ["bottom_right", "bottom_left", "top_right", "top_left", "center", "bottom_center", "top_center", "tile"]},
            "opacity": {"type": "number", "description": "0..1, default 0.6"},
            "scale": {"type": "number", "description": "watermark width as fraction of image width, default 0.22"},
            "color": {"type": "string", "description": "text colour hex, default #FFFFFF"},
            "preset": {"type": "string", "enum": ["disclosure"], "description": "'disclosure' = MLS-style 'Digitally altered' label (bottom-left, small). Use when the customer asks for an altered/edited/virtually-staged disclosure."},
        }, "required": ["photo_ids"]},
    },
    {
        "name": "remove_object",
        "description": "Remove objects with a precise segmentation mask and fill the hole (eraser for small objects, generative fill for large ones). Only the masked area changes. Photos where the target is not found are left unchanged and reported with found=false.",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "target": {"type": "string", "description": "short noun phrase for what to remove, e.g. 'cars', 'blue wheelie bin', 'power cord and plug', 'lockbox'"},
            "method": {"type": "string", "enum": ["auto", "erase", "generative"], "description": "auto picks by mask size"},
            "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                       "description": "optional normalized points the user placed on the object (single photo only)"},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                    "description": "optional normalized box to restrict where to look (single photo only)"},
            "fill_prompt": {"type": "string", "description": "optional description of what should be behind the object, e.g. 'concrete driveway and white garage door'"},
            "painted_area": {"type": "boolean", "description": "include the region the user painted on this photo (see painted_area_by_user in the state); single photo only"},
        }, "required": ["photo_ids", "target"]},
    },
    {
        "name": "generative_edit",
        "description": "Open-ended photorealistic edit from an instruction (turn on lamps, put a reference image on the TV, iron the bedding, regenerate the lawn, add blinds, open the pool, sharpen). The result is composited back into the original: set `protect` to the object being changed so nothing else moves.",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "instruction": {"type": "string", "description": "precise instruction for the image model"},
            "references": {"type": "array", "items": {"type": "string"},
                           "description": "optional reference images: 'reference:0', 'bracket:1', 'original'"},
            "protect": {"type": "string", "description": "'auto' (keep only changed regions), 'none' (accept whole frame), or a noun phrase to confine the edit to, e.g. 'lamp', 'tv screen', 'bed', 'lawn'"},
            "painted_area": {"type": "boolean", "description": "include the region the user painted on this photo (see painted_area_by_user in the state); single photo only"},
        }, "required": ["photo_ids", "instruction"]},
    },
    {
        "name": "tv_screen",
        "description": "TV screen replacement: perspective-fit the customer's reference image onto the TV panel (mode 'reference'), or black the screen out (mode 'blackout'). Deterministic and exact - prefer this over generative_edit whenever a reference must appear on a TV. Photos with no TV are skipped.",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "mode": {"type": "string", "enum": ["reference", "blackout"]},
            "reference_index": {"type": "integer", "description": "which customer reference to use (0-based, default 0)"},
            "brightness": {"type": "number", "description": "0.6..1.0 screen brightness multiplier, default 0.92"},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "optional normalized box around the TV (single photo only)"},
        }, "required": ["photo_ids"]},
    },
    {
        "name": "blur",
        "description": "Privacy blur / pixelate something (faces in framed photos, people, licence plates, house numbers, screens, documents). Only the detected regions are blurred.",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "target": {"type": "string", "description": "what to blur, e.g. 'faces in the framed photos', 'licence plate', 'house number'"},
            "strength": {"type": "number", "description": "0..100, default 60"},
            "pixelate": {"type": "boolean", "description": "mosaic instead of gaussian blur"},
            "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        }, "required": ["photo_ids", "target"]},
    },
    {
        "name": "match_look",
        "description": "Make photos consistent with one reference photo of the same shoot: transfers overall colour balance, brightness and contrast (not content). Use for 'make them all match photo 3', 'same tone as the first one', 'these two look different'.",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "reference_photo_id": {"type": "string", "description": "the photo whose look the others should match"},
            "strength": {"type": "number", "description": "0..1, default 0.8"},
        }, "required": ["photo_ids", "reference_photo_id"]},
    },
    {
        "name": "crop_rotate",
        "description": "Straighten (rotate by a small angle, positive = counter-clockwise) and/or crop to a normalized box. The output keeps the frame size.",
        "input_schema": {"type": "object", "properties": {
            "photo_id": {"type": "string"},
            "rotate_deg": {"type": "number", "description": "e.g. 1.5 to level a tilted horizon"},
            "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4, "description": "normalized crop [x0,y0,x1,y1]; aspect is kept"},
        }, "required": ["photo_id"]},
    },
    {
        "name": "undo",
        "description": "Step photos back to their previous version.",
        "input_schema": {"type": "object", "properties": {
            "photo_ids": PHOTO_IDS,
            "steps": {"type": "integer", "description": "default 1"},
        }, "required": ["photo_ids"]},
    },
    {
        "name": "revert_to",
        "description": "Make a specific version the current one (0 = original finished photo).",
        "input_schema": {"type": "object", "properties": {
            "photo_id": {"type": "string"},
            "version_index": {"type": "integer"},
        }, "required": ["photo_id", "version_index"]},
    },
]
