UMDThesisTemplate
=================

UMD Computer Science Thesis Template

This is the UMD LaTeX style sheet and template.  Re-started by Pete
Willemsen and fixed up by Andrew Larson, Scot Halverson, and Matt Overby.

Getting Started
---------------

The files you will want to edit are the files ending in .tex. These
are LaTeX files and have been appropriately named to correspond to a
standard thesis document (abstract, chapters, appendices, etc...). The
primary file that bootstraps the creation of the PDF and combines all
the sub TeX files is __UMDCS_MSThesis.tex__. Your bibliography entries
will be placed in UMDCS_MSThesis.bib.

Creating the PDF and Running LaTeX
-----------------------------------

To create your PDF file, you will need to issue the following
commands:

1. xelatex UMDCS_MSThesis
2. biber UMDCS_MSThesis
3. xelatex UMDCS_MSThesis

The reason for the three steps is to first run LaTeX over your
document for the first pass at finding references. The biber command
will pull these references from your bibliography file and prepare
them for inclusion in your document. The final step brings everything
together and creates the UMDCS_MSThesis.pdf file. Note that if you
haven't modified your bibliography file (UMDCS_MSThesis.bib), then you
needn't re-run the first two steps each time.

## A Note on Images:
To include images, you will use the "includegraphics" macro or the
tikz package. However you will need to ensure your images are in one
of the supported formats: either pdf, jpg or png.



