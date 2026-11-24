"""Insert the two protocol sections immediately before chapters 6 and 7."""
import argparse
from pathlib import Path
import re
from copy import deepcopy
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def insert(source, protocol, output):
    doc=Document(source)
    sections=[s.strip().split("\n\n") for s in re.split(r"^## ",Path(protocol).read_text(),flags=re.M) if s.strip()]
    if len(sections)!=2:
        raise ValueError("expected exactly two study sections")
    for section,chapter in zip(sections,["六6６","七7７"]):
        if any(p.text.strip()==section[0] for p in doc.paragraphs):
            raise ValueError("section already present")
        choices=[p for p in doc.paragraphs if re.match(r"^第["+chapter+r"]章",re.sub(r"\s","",p.text))
                 and not re.search(r"\d\s*$",p.text.strip())]
        if not choices:
            raise ValueError(f"missing chapter {chapter}")
        anchor=choices[-1]
        # Copy a nearby actual body paragraph's properties to retain the draft typography.
        paras=doc.paragraphs
        i=next(i for i,p in enumerate(paras) if p._p is anchor._p)
        template=next((p for p in reversed(paras[:i]) if len(p.text.strip())>80),None)
        for j,body in enumerate(section):
            p=anchor.insert_paragraph_before()
            if template is not None and template._p.pPr is not None:
                properties=deepcopy(template._p.pPr)
                for name in ["sectPr","pageBreakBefore","numPr"]:
                    for item in properties.findall(qn("w:"+name)):
                        properties.remove(item)
                p._p.insert(0,properties)
            run=p.add_run(body.strip())
            if j==0:
                run.bold=True
                p.paragraph_format.keep_with_next=True
    settings=doc.settings.element
    update=settings.find(qn("w:updateFields"))
    if update is None:
        update=OxmlElement("w:updateFields"); settings.append(update)
    update.set(qn("w:val"),"true")
    doc.save(output)
    reopened=Document(output)
    for s in sections:
        assert sum(p.text==s[0] for p in reopened.paragraphs)==1


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for key in ["source","protocol","output"]:
        ap.add_argument(key)
    a=ap.parse_args()
    insert(a.source,a.protocol,a.output)
