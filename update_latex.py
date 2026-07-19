import re

def update_latex():
    # Read the tex file for content
    try:
        with open('checkmate_ieee_paper.tex', 'r', encoding='utf-8') as f:
            tex_content = f.read()
    except Exception as e:
        print(f"Error reading tex file: {e}")
        return
    
    # Read the latex report file
    try:
        with open('latexreport', 'r', encoding='utf-8') as f:
            latex_content = f.read()
    except Exception as e:
        print(f"Error reading latexreport file: {e}")
        return

    # Extract sections from checkmate_ieee_paper.tex
    # For IEEE papers, sections usually start with \section{Title}
    
    # Let's extract the abstract first
    abs_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex_content, re.DOTALL | re.IGNORECASE)
    abstract = abs_match.group(1).strip() if abs_match else ""
    
    # Replace abstract in latexreport
    latex_content = re.sub(
        r'(\\chapter\*\{Abstract\}.*?\\addcontentsline\{toc\}\{chapter\}\{Abstract\}).*?(\\chapter\{Introduction\})',
        r'\g<1>\n\n' + abstract + r'\n\n\g<2>',
        latex_content,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Now let's extract sections and convert to chapters/sections for the report
    sections_match = re.findall(r'\\section\{(.*?)\}(.*?)(?=\\section\{|\\end\{document\}|\\bibliographystyle)', tex_content, re.DOTALL | re.IGNORECASE)
    
    latex_body = []
    
    for title, content in sections_match:
        if title.lower() == 'introduction':
            latex_body.append(f"\\chapter{{{title}}}")
        elif title.lower() in ['literature review', 'system architecture', 'methodology', 'implementation', 'results and discussion', 'conclusion']:
            latex_body.append(f"\\chapter{{{title}}}")
        else:
            latex_body.append(f"\\chapter{{{title}}}")
            
        # we also need to convert subsections
        # checkmate_ieee_paper.tex uses \subsection{}
        content = re.sub(r'\\subsection\{(.*?)\}', r'\\section{\1}', content)
        content = re.sub(r'\\subsubsection\{(.*?)\}', r'\\subsection{\1}', content)
        
        latex_body.append(content.strip())
        latex_body.append("\n")
        
    formatted_body = "\n".join(latex_body)
    
    # Replace the body of the report
    start_match = re.search(r'\\chapter\{Introduction\}', latex_content, re.IGNORECASE)
    end_match = re.search(r'\\bibliographystyle|\\bibliographystyle', latex_content, re.IGNORECASE)
    
    if start_match and end_match:
        new_latex = latex_content[:start_match.start()] + formatted_body + "\n\n" + latex_content[end_match.start():]
        with open('latexreport', 'w', encoding='utf-8') as f:
            f.write(new_latex)
        print("Updated latexreport successfully from checkmate_ieee_paper.tex")
    else:
        print("Could not find start or end markers in latexreport for the body. Attempting a broader replace.")

if __name__ == '__main__':
    update_latex()

