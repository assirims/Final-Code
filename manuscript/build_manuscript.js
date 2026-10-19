// Builds the manuscript (.docx, MDPI layout) from results/paper_numbers.json and results/figures.
// Usage: node manuscript/build_manuscript.js [output.docx]
//   HIGHLIGHT=1 node manuscript/build_manuscript.js  -> revision with new/changed text highlighted in yellow
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, Footer, PageNumber, Header } = require("docx");
const L = require("./lib");
const ROOT = path.resolve(__dirname, "..");
const N = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "paper_numbers.json"), "utf8"));
const FIG = path.join(ROOT, "results", "figures");
const F = { path: (f) => path.join(FIG, f) };

const TITLE = "An IoT-Based Healthcare Monitoring and Surveillance Framework for People with Special Needs";
const front = require("./content_front")(N, F, TITLE);
const intro = require("./content_intro")(N, F);
const methods = require("./content_methods")(N, F);
const results = [...require("./content_results")(N, F), ...require("./content_results2")(N, F)];
const discussion = require("./content_discussion")(N, F);
const back = require("./content_back")(N);
const refsTitle = L.H1("References");
const refList = L.ReferenceList();

const doc = new Document({
  creator: "Anas Bilal", title: TITLE, description: "Manuscript (MDPI layout)",
  styles: { default: { document: { run: { font: L.FONT, size: L.BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 20, bold: true, font: L.FONT }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 20, italics: true, font: L.FONT }, paragraph: { spacing: { before: 140, after: 60 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 20, italics: true, font: L.FONT }, paragraph: { spacing: { before: 100, after: 40 }, outlineLevel: 2 } },
    ] },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: L.PAGE_W, height: 16838 }, margin: { top: 1134, bottom: 1134, left: L.MARGIN, right: L.MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "IoT-CARES: IoT-based monitoring and surveillance for people with special needs", font: L.FONT, size: 14, italics: true, color: "666666" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: L.FONT, size: 16 })] })] }) },
    children: [...front, ...intro, ...methods, ...results, ...discussion, ...back, refsTitle, ...refList],
  }],
});
const out = process.argv[2] || path.join(ROOT, "manuscript", L.HL ? "IoT-CARES_Revised_HIGHLIGHTED.docx" : "IoT-CARES_Revised_Clean.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out, "figures:", L.figNum(), "tables:", L.tabNum()); });
