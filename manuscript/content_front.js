// Title block, abstract and keywords (MDPI layout). The abstract is generated from paper_numbers.json.
const L = require("./lib");
const { Paragraph, TextRun, AlignmentType } = require("docx");
module.exports = function (N, F, TITLE) {
  const out = [];
  out.push(new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Article", italics: true, font: L.FONT, size: 20 })] }));
  out.push(new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: TITLE, bold: true, font: L.FONT, size: 36 })] }));
  out.push(new Paragraph({ spacing: { after: 100 }, children: [
    new TextRun({ text: "Anas Bilal", bold: true, font: L.FONT, size: 20 }), new TextRun({ text: "1,*", bold: true, superScript: true, font: L.FONT, size: 20 }),
    new TextRun({ text: " and [Co-Author Name]", bold: true, font: L.FONT, size: 20 }), new TextRun({ text: "2", bold: true, superScript: true, font: L.FONT, size: 20 })] }));
  out.push(new Paragraph({ spacing: { after: 20 }, children: L.runs("^1^ [Department], [Institution], [City] [Postcode], [Country]", { size: 16 }) }));
  out.push(new Paragraph({ spacing: { after: 20 }, children: L.runs("^2^ [Department], [Institution], [City] [Postcode], [Country]", { size: 16 }) }));
  out.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "* Correspondence: a.bilal19@yahoo.com", font: L.FONT, size: 16 })] }));
  const abs = require("./abstract")(N);
  out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 260 }, children: [new TextRun({ text: "Abstract: ", bold: true, font: L.FONT, size: 18 }), ...L.runs(abs, { size: 18 })] }));
  out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 240 }, children: [new TextRun({ text: "Keywords: ", bold: true, font: L.FONT, size: 18 }), ...L.runs("Internet of Things; remote health monitoring; people with disabilities; older adults; fall detection; freezing of gait; arrhythmia; dementia; federated learning; conformal prediction; edge computing; explainable artificial intelligence", { size: 18 })] }));
  return out;
};
