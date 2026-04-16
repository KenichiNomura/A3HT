#!/usr/bin/env python3
"""Generate a simple PowerPoint summary for glassy-carbon analysis outputs."""

import csv
import json
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

EMU_PER_INCH = 914400
SLIDE_W = 10 * EMU_PER_INCH
SLIDE_H = int(7.5 * EMU_PER_INCH)
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value):
    return "%.1f%%" % (100.0 * float(value))


def f3(value):
    return "%.3f" % float(value)


def format_coord_hist(hist):
    return ", ".join("C%d: %d" % (int(x["coordination"]), int(x["count"])) for x in hist)


def format_ring_hist(hist):
    return ", ".join("%d-member: %d" % (int(x["ring_size"]), int(x["bond_count"])) for x in hist)


def text_run(text, size=1800, bold=False, color="1F2937"):
    attrs = ['lang="en-US"', 'sz="%d"' % size, 'dirty="0"']
    if bold:
        attrs.append('b="1"')
    return (
        "<a:r>"
        '<a:rPr %s><a:solidFill><a:srgbClr val="%s"/></a:solidFill></a:rPr>'
        "<a:t>%s</a:t>"
        "</a:r>"
    ) % (" ".join(attrs), color, escape(text))


def paragraph(text, level=0, size=1800, bold=False, bullet=False, color="1F2937"):
    bullet_xml = '<a:buChar char="•"/>' if bullet else '<a:buNone/>'
    return '<a:p><a:pPr lvl="%d"/>%s%s<a:endParaRPr lang="en-US" sz="%d"/></a:p>' % (
        level,
        bullet_xml,
        text_run(text, size=size, bold=bold, color=color),
        size,
    )


def text_box(shape_id, x, y, w, h, paragraphs_xml):
    x_emu = int(x * EMU_PER_INCH)
    y_emu = int(y * EMU_PER_INCH)
    w_emu = int(w * EMU_PER_INCH)
    h_emu = int(h * EMU_PER_INCH)
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="%d" name="TextBox %d"/><p:cNvSpPr txBox="1"/><p:nvPr/>'
        '</p:nvSpPr><p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        '<p:txBody><a:bodyPr wrap="square" rtlCol="0" anchor="t"/><a:lstStyle/>%s</p:txBody></p:sp>'
    ) % (shape_id, shape_id, x_emu, y_emu, w_emu, h_emu, "".join(paragraphs_xml))


def title_shape(shape_id, text):
    return text_box(shape_id, 0.5, 0.3, 9.0, 0.6, [paragraph(text, size=2800, bold=True, color="0B3954")])


def slide_xml(title, body_shapes_xml):
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:cSld><p:spTree>'
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        '%s%s</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
    ) % (NS_A, NS_R, NS_P, title_shape(2, title), "".join(body_shapes_xml))


def slide_rels_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>'


def content_types_xml(slide_count):
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for slide_idx in range(1, slide_count + 1):
        overrides.append('<Override PartName="/ppt/slides/slide%d.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' % slide_idx)
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>%s</Types>' % ''.join(overrides)


def root_rels_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>'


def app_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>OpenAI Codex</Application><PresentationFormat>On-screen Show (4:3)</PresentationFormat><Slides>5</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides><MMClips>0</MMClips><ScaleCrop>false</ScaleCrop><HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Theme</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs><TitlesOfParts><vt:vector size="6" baseType="lpstr"><vt:lpstr>Office Theme</vt:lpstr><vt:lpstr>Slide 1</vt:lpstr><vt:lpstr>Slide 2</vt:lpstr><vt:lpstr>Slide 3</vt:lpstr><vt:lpstr>Slide 4</vt:lpstr><vt:lpstr>Slide 5</vt:lpstr></vt:vector></TitlesOfParts><AppVersion>1.0</AppVersion></Properties>'


def core_xml():
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Glassy Carbon Analysis Summary</dc:title><dc:creator>OpenAI Codex</dc:creator><cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">%s</dcterms:modified></cp:coreProperties>' % (now, now)


def presentation_xml(slide_count):
    sld_ids = ''.join('<p:sldId id="%d" r:id="rId%d"/>' % (255 + idx, idx + 1) for idx in range(1, slide_count + 1))
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="%s" xmlns:r="%s" xmlns:p="%s" saveSubsetFonts="1" autoCompressPictures="0"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>%s</p:sldIdLst><p:sldSz cx="%d" cy="%d"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>' % (NS_A, NS_R, NS_P, sld_ids, SLIDE_W, SLIDE_H)


def presentation_rels_xml(slide_count):
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for idx in range(1, slide_count + 1):
        rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide%d.xml"/>' % (idx + 1, idx))
    rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>' % (slide_count + 2))
    rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>' % (slide_count + 3))
    rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>' % (slide_count + 4))
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">%s</Relationships>' % ''.join(rels)


def pres_props_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:extLst/></p:presentationPr>' % (NS_A, NS_R, NS_P)


def view_props_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:normalViewPr/><p:slideViewPr><p:cSldViewPr snapToGrid="0" snapToObjects="1"/></p:slideViewPr><p:notesTextViewPr/></p:viewPr>' % (NS_A, NS_R, NS_P)


def table_styles_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="%s" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>' % NS_A


def theme_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="%s" name="Codex Theme"><a:themeElements><a:clrScheme name="Codex Colors"><a:dk1><a:srgbClr val="1F2937"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="0B3954"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="0EA5E9"/></a:accent1><a:accent2><a:srgbClr val="10B981"/></a:accent2><a:accent3><a:srgbClr val="F59E0B"/></a:accent3><a:accent4><a:srgbClr val="EF4444"/></a:accent4><a:accent5><a:srgbClr val="8B5CF6"/></a:accent5><a:accent6><a:srgbClr val="14B8A6"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme><a:fontScheme name="Codex Fonts"><a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Codex Format"><a:fillStyleLst><a:solidFill><a:schemeClr val="lt1"/></a:solidFill><a:solidFill><a:schemeClr val="lt2"/></a:solidFill><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="dk1"/></a:solidFill></a:ln><a:ln w="25400"><a:solidFill><a:schemeClr val="dk1"/></a:solidFill></a:ln><a:ln w="38100"><a:solidFill><a:schemeClr val="dk1"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle/><a:effectStyle/><a:effectStyle/></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="lt1"/></a:solidFill><a:solidFill><a:schemeClr val="lt2"/></a:solidFill><a:solidFill><a:schemeClr val="dk1"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>' % NS_A


def slide_master_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:cSld name="Office Theme"><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>' % (NS_A, NS_R, NS_P)


def slide_master_rels_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>'


def slide_layout_xml():
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="%s" xmlns:r="%s" xmlns:p="%s" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>' % (NS_A, NS_R, NS_P)


def build_slides(anneal, nemd, trajectory_rows):
    first = trajectory_rows[0]
    mid = trajectory_rows[len(trajectory_rows) // 2]
    last = trajectory_rows[-1]
    delta_density = float(nemd['density_g_cm3']) - float(anneal['density_g_cm3'])
    delta_sp2 = float(nemd['sp2_like_fraction']) - float(anneal['sp2_like_fraction'])
    delta_planarity = float(nemd['threefold_planarity_rms_mean_angstrom']) - float(anneal['threefold_planarity_rms_mean_angstrom'])
    return [
        slide_xml('Glassy Carbon Analysis Summary', [
            text_box(3, 0.7, 1.3, 8.6, 2.0, [
                paragraph('Run: my_runs/100', size=2200, bold=True, color='0F172A'),
                paragraph('Data sources: annealed snapshot, annealing trajectory, and NEMD final frame', size=1800),
                paragraph('Atoms: %d   Box: 20 x 20 x 40 A' % int(anneal['atom_count']), size=1800),
                paragraph('Bond graph cutoff: %.2f A' % float(anneal['bond_cutoff_angstrom']), size=1800),
            ]),
            text_box(4, 0.7, 3.6, 8.8, 2.6, [
                paragraph('Headline result', size=2200, bold=True, color='0B3954'),
                paragraph('The annealing trajectory shifts the structure toward predominantly 3-coordinated carbon with improved local planarity.', bullet=True),
                paragraph('The NEMD final frame is slightly denser and more graphitic than the annealed snapshot by the same cutoff-based metrics.', bullet=True),
            ]),
            text_box(5, 0.7, 6.5, 8.8, 0.5, [paragraph('Generated from analysis outputs under my_runs/100/analysis', size=1400, color='475569')]),
        ]),
        slide_xml('Annealed Snapshot', [
            text_box(3, 0.7, 1.1, 4.0, 5.6, [
                paragraph('Key metrics', size=2200, bold=True, color='0B3954'),
                paragraph('Density: %s g/cm^3' % f3(anneal['density_g_cm3']), bullet=True),
                paragraph('Mean coordination: %s' % f3(anneal['mean_coordination']), bullet=True),
                paragraph('sp2-like fraction: %s' % pct(anneal['sp2_like_fraction']), bullet=True),
                paragraph('sp3-like fraction: %s' % pct(anneal['sp3_like_fraction']), bullet=True),
                paragraph('Undercoordinated fraction: %s' % pct(anneal['undercoordinated_fraction']), bullet=True),
                paragraph('Mean bond length: %s A' % f3(anneal['bond_length_mean_angstrom']), bullet=True),
                paragraph('Mean bond angle: %.2f deg' % float(anneal['bond_angle_mean_deg']), bullet=True),
                paragraph('3-fold planarity RMS: %s A' % f3(anneal['threefold_planarity_rms_mean_angstrom']), bullet=True),
            ]),
            text_box(4, 4.9, 1.1, 4.2, 5.6, [
                paragraph('Topology summary', size=2200, bold=True, color='0B3954'),
                paragraph('Coordination counts: %s' % format_coord_hist(anneal['coordination_histogram']), bullet=True),
                paragraph('Ring proxy: %s' % format_ring_hist(anneal['ring_proxy_bond_histogram']), bullet=True),
                paragraph('Interpretation: a strongly 3-coordinated network with dominant 5- and 6-member character.', bullet=True),
                paragraph('Residual 2-coordinated sites indicate edge-like or dangling-bond environments in the disordered network.', bullet=True),
            ]),
        ]),
        slide_xml('Annealing Trajectory', [
            text_box(3, 0.7, 1.1, 8.6, 1.0, [paragraph('90 frames analyzed from timestep 5,000 to 450,000', size=2200, bold=True, color='0B3954')]),
            text_box(4, 0.7, 2.0, 4.1, 4.6, [
                paragraph('Evolution of key metrics', size=2200, bold=True, color='0B3954'),
                paragraph('sp2-like fraction: %s -> %s' % (pct(first['sp2_like_fraction']), pct(last['sp2_like_fraction'])), bullet=True),
                paragraph('sp3-like fraction: %s -> %s' % (pct(first['sp3_like_fraction']), pct(last['sp3_like_fraction'])), bullet=True),
                paragraph('Mean coordination: %s -> %s' % (f3(first['mean_coordination']), f3(last['mean_coordination'])), bullet=True),
                paragraph('3-fold planarity RMS: %s A -> %s A' % (f3(first['threefold_planarity_rms_mean_angstrom']), f3(last['threefold_planarity_rms_mean_angstrom'])), bullet=True),
                paragraph('Density remains constant because the annealing trajectory used a fixed box.', bullet=True),
            ]),
            text_box(5, 5.0, 2.0, 4.3, 4.6, [
                paragraph('Representative frames', size=2200, bold=True, color='0B3954'),
                paragraph('Start  (t = %s): sp2 %s, sp3 %s, planarity %s A' % (first['timestep'].replace('.0', ''), pct(first['sp2_like_fraction']), pct(first['sp3_like_fraction']), f3(first['threefold_planarity_rms_mean_angstrom'])), bullet=True),
                paragraph('Middle (t = %s): sp2 %s, sp3 %s, planarity %s A' % (mid['timestep'].replace('.0', ''), pct(mid['sp2_like_fraction']), pct(mid['sp3_like_fraction']), f3(mid['threefold_planarity_rms_mean_angstrom'])), bullet=True),
                paragraph('End    (t = %s): sp2 %s, sp3 %s, planarity %s A' % (last['timestep'].replace('.0', ''), pct(last['sp2_like_fraction']), pct(last['sp3_like_fraction']), f3(last['threefold_planarity_rms_mean_angstrom'])), bullet=True),
                paragraph('The coordination log and structural metrics agree: n3 rises while n2 and n4 decline overall.', bullet=True),
            ]),
        ]),
        slide_xml('Anneal vs NEMD Final Frame', [
            text_box(3, 0.7, 1.1, 4.1, 5.5, [
                paragraph('Annealed snapshot', size=2200, bold=True, color='0B3954'),
                paragraph('Density: %s g/cm^3' % f3(anneal['density_g_cm3']), bullet=True),
                paragraph('Mean coordination: %s' % f3(anneal['mean_coordination']), bullet=True),
                paragraph('sp2-like fraction: %s' % pct(anneal['sp2_like_fraction']), bullet=True),
                paragraph('sp3-like fraction: %s' % pct(anneal['sp3_like_fraction']), bullet=True),
                paragraph('Bond length mean: %s A' % f3(anneal['bond_length_mean_angstrom']), bullet=True),
                paragraph('Planarity RMS: %s A' % f3(anneal['threefold_planarity_rms_mean_angstrom']), bullet=True),
            ]),
            text_box(4, 5.1, 1.1, 4.1, 5.5, [
                paragraph('NEMD final frame', size=2200, bold=True, color='0B3954'),
                paragraph('Density: %s g/cm^3' % f3(nemd['density_g_cm3']), bullet=True),
                paragraph('Mean coordination: %s' % f3(nemd['mean_coordination']), bullet=True),
                paragraph('sp2-like fraction: %s' % pct(nemd['sp2_like_fraction']), bullet=True),
                paragraph('sp3-like fraction: %s' % pct(nemd['sp3_like_fraction']), bullet=True),
                paragraph('Bond length mean: %s A' % f3(nemd['bond_length_mean_angstrom']), bullet=True),
                paragraph('Planarity RMS: %s A' % f3(nemd['threefold_planarity_rms_mean_angstrom']), bullet=True),
            ]),
            text_box(5, 0.7, 6.1, 8.6, 0.8, [paragraph('Change to NEMD final frame: density %+0.3f g/cm^3, sp2 %+0.1f points, planarity %+0.3f A' % (delta_density, 100.0 * delta_sp2, delta_planarity), size=1800, bold=True, color='7C2D12')]),
        ]),
        slide_xml('Conclusions And Files', [
            text_box(3, 0.7, 1.1, 8.7, 4.8, [
                paragraph('Takeaways', size=2200, bold=True, color='0B3954'),
                paragraph('Annealing drives the network toward a mostly 3-coordinated carbon structure with reduced nonplanarity.', bullet=True),
                paragraph('The final annealed state still retains a measurable 2-coordinated population, so the structure is not fully graphitic.', bullet=True),
                paragraph('The NEMD final frame appears slightly denser and more ordered by bond-length spread, planarity, and normal-alignment metrics.', bullet=True),
                paragraph('The ring proxy is dominated by 5- and 6-member environments, consistent with curved and disordered graphitic motifs.', bullet=True),
                paragraph('All underlying analysis tables and plots are preserved under my_runs/100/analysis for reuse.', bullet=True),
            ]),
            text_box(4, 0.7, 6.0, 8.7, 0.9, [paragraph('Key files: anneal/summary.json, nemd/summary.json, anneal_timeseries/trajectory_summary.csv', size=1500, color='475569')]),
        ]),
    ]


def write_pptx(output_path, slides):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(output_path), 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types_xml(len(slides)))
        zf.writestr('_rels/.rels', root_rels_xml())
        zf.writestr('docProps/app.xml', app_xml())
        zf.writestr('docProps/core.xml', core_xml())
        zf.writestr('ppt/presentation.xml', presentation_xml(len(slides)))
        zf.writestr('ppt/_rels/presentation.xml.rels', presentation_rels_xml(len(slides)))
        zf.writestr('ppt/presProps.xml', pres_props_xml())
        zf.writestr('ppt/viewProps.xml', view_props_xml())
        zf.writestr('ppt/tableStyles.xml', table_styles_xml())
        zf.writestr('ppt/theme/theme1.xml', theme_xml())
        zf.writestr('ppt/slideMasters/slideMaster1.xml', slide_master_xml())
        zf.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels', slide_master_rels_xml())
        zf.writestr('ppt/slideLayouts/slideLayout1.xml', slide_layout_xml())
        for idx, slide in enumerate(slides, start=1):
            zf.writestr('ppt/slides/slide%d.xml' % idx, slide)
            zf.writestr('ppt/slides/_rels/slide%d.xml.rels' % idx, slide_rels_xml())


def main():
    base = Path('my_runs/100/analysis')
    anneal = read_json(base / 'anneal' / 'summary.json')
    nemd = read_json(base / 'nemd' / 'summary.json')
    trajectory_rows = read_csv_rows(base / 'anneal_timeseries' / 'trajectory_summary.csv')
    slides = build_slides(anneal, nemd, trajectory_rows)
    output_path = base / 'glassy_carbon_analysis_summary.pptx'
    write_pptx(output_path, slides)
    print(output_path)


if __name__ == '__main__':
    main()
