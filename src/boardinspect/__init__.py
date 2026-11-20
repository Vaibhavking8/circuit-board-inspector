"""BoardInspect — offline desktop web app that finds missing components on PCBs.

The package is layered so the computer-vision code (imaging, registration,
comparison, decision) is pure and web-free; it is orchestrated by
``pipeline`` and exposed through the ``api`` package. See the project's
``ARCHITECTURE.md`` for the full layering rules.
"""

__version__ = "0.1.0"
