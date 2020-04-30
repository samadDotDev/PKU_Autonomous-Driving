% -*- mode: LaTeX; -*-
%
% ------------------------ identification -----------------------------
% Originally created by Paul Henning <paul-henning@uiowa.edu>
%
\NeedsTeXFormat{LaTeX2e}[1995/06/01]
\ProvidesClass{umdthesis}[2014/12/01 Univ. of MN Duluth Thesis Class]
% ChangeLog
%
% Wed Apr 15 17:28:07 CDT 2009	Andrew Larson <lars2865@d.umn.edu>
%    * Adjusted the margins, still using the old uit labels.
%		 * Modified the title, signature, copyright, dedication and
%      acknowledgement pages to fit UMD MS Thesis samples.
%
% Tue Apr  14 10:12:07 CDT 2009  Pete Willemsen <willemsn@d.umn.edu>
%    * Resurrected the thesis class for use here at UMD.
%
% Nov 24 2014 Pete Willemsen <willemsn@d.umn.edu>
% Brought this to current LaTeX (http://www.olivierverdier.com/posts/2013/07/15/modern-latex/).
% ------------------------- initial code ------------------------------
\RequirePackage{ifthen} 
\newboolean{nofrontmatter} % Skip everything before the index
\newboolean{nofigures} % There are no figures (no list of figures)
\newboolean{notables} % There are no tables (no list of tables)
\newboolean{appendix} % There is an appendix [N/A]
\newboolean{ackpage} % Not an option anymore!
\newboolean{abstractpage} % There is an abstract page 
\newboolean{copyrightpage} % There is a copyright page
\newboolean{dedicationpage} % There is a dedication page
% --------------------- declaration of options ------------------------
\DeclareOption{nofrontmatter}{\setboolean{nofrontmatter}{true}}
\DeclareOption{nofigures}{\setboolean{nofigures}{false}}
\DeclareOption{notables}{\setboolean{notables}{true}}
\DeclareOption{appendix}{\setboolean{appendix}{true}}
\DeclareOption{abstractpage}{\setboolean{abstractpage}{true}}
\DeclareOption{copyrightpage}{\setboolean{copyrightpage}{true}}
\DeclareOption{dedicationpage}{\setboolean{dedicationpage}{true}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
% ------------------------ option parsing -----------------------------
\ProcessOptions
% --------------------- package loading -------------------------------
\LoadClass[12pt]{report}
% \RequirePackage[utf8]{inputenc}  % not needed when xelatex is used to create the PDF
\RequirePackage{tikz}
\RequirePackage{fancyhdr,indentfirst}
\RequirePackage[margin=1in,left=1.5in, bottom=1.5in]{geometry}
% \RequirePackage[pdftex]{graphicx} 
\RequirePackage{setspace}
\RequirePackage[pdfusetitle,colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\RequirePackage{listings}
\lstset{language=C++}  %% You can change the language if you like too
\lstset{tabsize=2}
\lstset{basicstyle=\normalsize}
\usepackage{xcolor}
\usepackage{color}

% --------------------------  main code -------------------------------
%%% PREAMBLE COMMANDS
\newcommand\advisor{\gdef\@advisor}
\def\submitdate#1{\gdef\@submitdate{#1}}
\def\copyrightyear#1{\setboolean{copyrightpage}{true}\gdef\@copyrightyear{#1}}
\def\dedication#1{\setboolean{dedicationpage}{true}\gdef\@dedication{#1}}
\def\ackfile#1{\setboolean{ackpage}{true}\gdef\@ackfile{#1}}
\def\abstractfile#1{\setboolean{abstractpage}{true}\gdef\@abstractfile{#1}}

\def\@title{}
\def\@author{}
\def\@submitdate{\ifcase\the\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space \number\the\year}
\ifnum\month=12
    \@tempcnta=\year \advance\@tempcnta by 1
    \edef\@copyrightyear{\number\the\@tempcnta}
\else
    \def\@copyrightyear{\number\the\year}
\fi

\newlength{\li}\setlength{\li}{14.48pt}



% This sets up the page numbering for the prefix material
\newcommand{\umdtprefixpg}
{
	\pagenumbering{roman}
	\lhead{}\chead{}\rhead{}
	\lfoot{}\cfoot{\fancyplain{\thepage}{\thepage}}\rfoot{}
}


% This sets up the page numbering for text and references
\newcommand{\umdtnormalpg}
{
	\pagenumbering{arabic}
	\lhead{}\chead{}\rhead{}
	\lfoot{}\cfoot{}\rfoot{\fancyplain{\thepage}{\thepage}}
}



% Kills the display of Chapter in "Chapter 1 \\ Title"
\renewcommand{\chaptername}{}

% Changes the chapter heading to something more basic
\renewcommand{\@makechapterhead}[1]
{
	\vspace*{.25in}
  {
    \parindent 	% Needed.
    \z@					% Font style.
    \huge \textbf{\thechapter} \hspace{.25in} \textbf{#1}
    \par
    \vskip 2\li
  }
}
  
% These renews are for figures to make them "nicer" - I guess.
\renewcommand\floatpagefraction{0.9}
\renewcommand\topfraction{0.9}
\renewcommand\bottomfraction{0.9}
\renewcommand\textfraction{.1}
\setcounter{topnumber}{4}
  

% This asks for all of the front matter for the thesis
\newcommand{\frontmatter}
{
	\ifthenelse{\boolean{nofrontmatter}}{}
	{
		\singlespace
                % Signature Page is no longer required/needed by the Grad School
		% \signaturepg   
		\titlepg
		\copyrightpg
		\umdtprefixpg
		\ackpg
		\dedicationpg
		\abstractpg
		\tableofcontents

		%\ifthenelse{\boolean{notables}}{}
		%{ % Tables?  
			% \listoftables \newpage
		%}
		\ifthenelse{\boolean{nofigures}}{}
		{ % Figures?
			\listoffigures \newpage
		}    
	}
	\raggedbottom
	\doublespace
	\umdtnormalpg
}


%
% TITLEPG - command to add the title page to the thesis
%
\newcommand{\titlepg}
{
	\thispagestyle{empty}
	\singlespace
	
	\vspace*{\stretch{1}}
	
	\begin{center}
		\@title
	\end{center}
	\vspace*{\stretch{2}}
	
	\begin{center}
		A PROJECT WRITEUP \\
		BY
	\end{center}
	\vspace*{\stretch{2}}
	
	\begin{center}
		\@author
	\end{center}
	\vspace*{\stretch{2}}

	\begin{center}
	%	IN FULFILLMENT OF THE REQUIREMENTS \\
	%	FOR THE SEMESTER PROJECT OF \\
	%	CS 5232 (FALL 2019) \\
	%	BY
	\end{center}
	\vskip 2\li 
	
	\begin{center}
	%	\@advisor
	\end{center}
	\vskip 2\li
	
	\begin{center}
		\@submitdate 
	\end{center}
	
	\vfill
	\newpage
}

%
% COPYRIGHTPG - command to add the copyright page to the thesis
%
\newcommand{\copyrightpg}
{
  \ifthenelse{\boolean{copyrightpage}}
  {
		\thispagestyle{empty}
		\vspace*{\fill}
		\begin{center}
			\copyright \hspace*{2mm} \@author \hspace*{2mm} \@copyrightyear
		\end{center}
		\vspace*{\fill}
		\newpage
  }
  {}
}


%
% DEDICATIONPG - command to add the dedication page to the thesis
%
\newcommand{\dedicationpg}
{
	\ifthenelse{\boolean{dedicationpage}}
	{
		\begin{center}
			Dedication
		\end{center}
		\input{\@dedication}
		\vfill
		\newpage
	}
	{}
}


% 
% ACKPG - command for setting up the acknowledgement page
%
\newcommand{\ackpg}
{
	\ifthenelse{\boolean{ackpage}}
	{
		\begin{center}
			Acknowledgements
		\end{center}
		\doublespace
		\input{\@ackfile}
		\vfill
		\newpage
	}
        {}
}


%
% ABSTRACTPG - command to add the abstract page
%
% ABSTRACT BOUND WITH THESIS (optional)
%
\newcommand{\abstractpg}
{ 
	\ifthenelse{\boolean{abstractpage}}
	{ 
		\begin{center}
			Abstract
		\end{center}
		\doublespace
		\input{\@abstractfile}
		\vfill
		\newpage
	}
	{}
}


%
% SIGNATURE PAGE - 
%
\newcommand{\signaturepg}
{
	\thispagestyle{empty}%
	
	\begin{center}
		UNIVERSITY OF MINNESOTA
	\end{center}
	\vspace*{\stretch{2}}
	
	\begin{center}
		This is to certify that I have examined this copy of a master's thesis by
	\end{center}
	\vspace*{\stretch{2}}
	
	\begin{center}
		\@author
	\end{center}
	\vspace*{\stretch{2}}
	
	\begin{center}
		and have found that it is complete and satisfactory in all respects,\\
		and that any and all revisions required by the final \\
		examining committee have been made.
	\end{center}
	\vspace*{\stretch{3}}
	
	\begin{center}
		\hrulefill \\
		Name of Faculty Advisor
	\end{center}
	\vspace*{\stretch{3}}
	
	\begin{center}
		\hrulefill \\
		Signature of Faculty Advisor
	\end{center}
	\vspace*{\stretch{3}}
	
	\begin{center}
		\hrulefill \\
		Date
	\end{center}
	\vspace*{\stretch{3}}
	
	\begin{center}
		GRADUATE SCHOOL
	\end{center}
	
	\vfill
	\newpage
}

