# Missing Part Detector

Offline visual inspection tool that finds missing components on circuit boards.

A fixed camera photographs every board. You show the app one good board,
name each part once, and from then on it tells you which named parts are
missing in new photos, and where. If nothing is missing, it shows a green **OK**.

> Status: work in progress 

## How it works

1. Upload a photo of a good board (the reference).
2. Draw a box around each part and give it a name.
3. Upload photos of the same board taken the same way.
4. For each photo, see missing parts highlighted in red, or a green **OK**.

Everything runs locally on your laptop. No internet, no cloud APIs.

## Assumptions

- Photos come from the same fixed camera, same resolution and viewpoint.
- Only small shifts and lighting changes between photos.

## Install and run

_Coming soon._

## Project docs

- [AI_COLLAB.md](AI_COLLAB.md): how I worked with AI and the decisions I made
- [RESEARCH.md](RESEARCH.md): what I looked into and why I chose this approach
- [ARCHITECTURE.md](ARCHITECTURE.md): system design and trade-offs
