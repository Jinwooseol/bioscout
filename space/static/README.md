---
title: BioScout
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: PubMed answers with every citation checked
---

# BioScout

A biomedical research agent that answers from PubMed, cites a paper for every sentence, and checks each
citation against the paper's abstract before showing it.

This Space is a static page: it shows answers recorded from the real agent for three example questions,
with the verdict of the citation check on every sentence. The page itself runs no model. When the
BioScout API is up, it also shows a question box, and a question typed there is sent to that API, which
answers a limited number of questions a day.

For the evaluation behind the numbers, and to run BioScout yourself, see the source:
https://github.com/Jinwooseol/bioscout

BioScout helps you find and read biomedical literature. It is not a medical device and must not be used
to make diagnostic or treatment decisions about individual patients.
