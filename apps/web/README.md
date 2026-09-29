# PIT Alpha Lab Web

The web application is a responsive React/TypeScript research surface over the project's read-only
FastAPI service. It never accepts filesystem paths and never mutates experiment artifacts.

## Local development

From the repository root, start the API with the isolated Python environment:

```powershell
.\.venv\Scripts\python.exe -m pitalpha serve --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
Set-Location apps\web
npm.cmd ci
npm.cmd run dev
```

Vite proxies `/api` to `127.0.0.1:8000`. Set `VITE_API_ROOT` only when the API is hosted at a
different origin. If the API cannot be reached, the interface enters a clearly marked audited
snapshot mode; live curves and run-level diagnostics remain disabled rather than fabricated.

## Checks

```powershell
npm.cmd run lint
npm.cmd run build
npm.cmd run smoke
```

The chart code is loaded as a separate chunk so the first screen does not wait for the visualization
library. The smoke check uses an existing local Edge installation and expects both development
servers to be running. Motion respects the operating system's reduced-motion preference.

CI uses `npm ci` against the committed lockfile and performs the same TypeScript plus production
build check. Browser screenshots and local profiles remain ignored research artifacts.
