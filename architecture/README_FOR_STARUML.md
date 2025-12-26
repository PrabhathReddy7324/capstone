System architecture (PlantUML) for the Aircraft Detection project

Files created:
- system_architecture.puml — PlantUML source showing component and sequence diagrams describing the project (frontend, backend, models, detectors, classifier, storage).

Options to view/import:

1) View with PlantUML (recommended)
   - Install PlantUML: https://plantuml.com/starting
   - From the repo root, run (if you have plantuml.jar):

     java -jar plantuml.jar architecture/system_architecture.puml

   - This generates a PNG/SVG of the diagram in the same folder.

2) Use a PlantUML plugin in your editor
   - VS Code: "PlantUML" extension can preview and export diagrams.
   - Open `architecture/system_architecture.puml` and view the preview.

3) Import into StarUML
   - StarUML does not natively open .puml, but you can:
     a) Export the PlantUML diagram to SVG or PNG (see option 1), then add the image to a StarUML diagram as an artifact or reference image.
     b) Or, use an external tool to convert PlantUML to XMI (rare) or recreate the diagram in StarUML using the visual editor.

If you want a native StarUML project file (.mdj) I can generate a simple skeleton, but StarUML's .mdj JSON structure is strict and regenerating an exact project file that imports cleanly is more reliable when made inside StarUML itself. Tell me if you'd like a generated .mdj skeleton and I'll produce one to the best approximation.

Notes about the diagram
- The component diagram shows how the React frontend (browser) talks to Flask backend (`/predict`).
- The backend uses YOLO for detection and a separate classifier for per-crop classification.
- Model files (YOLO and classifier checkpoints) reside on local filesystem; you can also host them in a network storage or object store.
- The sequence diagram shows the request lifecycle for a single uploaded image.

If you'd like, I can:
- Add more detail (DB, caching, Redis, message queue for async jobs).
- Create separate diagrams (deployment diagram, data-flow diagram, sequence for re-training pipeline).
- Generate a StarUML .mdj skeleton instead of PlantUML.
