"""Small LaTeX layout helpers for the manuscript's generated tables."""


def tabular(spec, header, rows):
    body = [r if r in (r"\midrule", r"\addlinespace") else r + r"\\" for r in rows]
    return "\n".join([r"\begin{tabular}{@{}" + spec + r"@{}}", r"\toprule",
                      header + r"\\", r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}"])


def panel(title, spec, header, rows, fit=False):
    body = tabular(spec, header, rows)
    if fit:
        body = (r"\sbox{\tablepanelbox}{" + body + r"}" +
                r"\ifdim\wd\tablepanelbox>\linewidth " +
                r"\resizebox{\linewidth}{!}{\usebox{\tablepanelbox}}" +
                r"\else\usebox{\tablepanelbox}\fi")
    return ((r"\par\smallskip\noindent\textbf{" + title + r"}\par\smallskip" + "\n") if title else "") + body


def note(text):
    return (r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize " +
            r"\textit{Note.} " + text + r"\end{minipage}")


def signed(value, digits=2):
    rounded = f"{value:+.{digits}f}"
    return f"{0:.{digits}f}" if float(rounded) == 0 else rounded


def scientific_p(value):
    if value >= 0.001:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.0e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"
