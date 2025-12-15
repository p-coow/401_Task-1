"""
Processor plugins for lab401.

Each processor module must expose a dict named PROCESSOR with keys:
- id: unique string identifier
- label: human-friendly name
- description: optional description
- process: callable taking (PIL.Image.Image, params: dict) -> PIL.Image.Image
"""