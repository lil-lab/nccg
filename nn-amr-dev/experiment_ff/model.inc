# Joint model resources
## Default lexical entry scorers
type=scorer.uniform id=uniform0Scorer   weight=0.0
type=scorer.uniform id=uniform1Scorer   weight=1.0
type=scorer.uniform id=uniform10Scorer  weight=10.0
type=scorer.lex.origin id=lexScorer default=uniform0Scorer scorers=genlex-txt:uniform1Scorer
type=scorer.lexeme.cooc id=coocScorer file=%{gizaProbs} lower=true
type=scorer.lexeme.origin id=lexemeScorer default=coocScorer scorers=genlex-align:uniform1Scorer,genlex-txt:uniform1Scorer
## Features
type=feat.logexp.coordination id=featLogexpCoordination cpp1=false cpap=false rept=true
type=feat.lex.amr id=featLex lexemeScorer=lexemeScorer entryScorer=lexScorer
type=feat.rule.shift id=featShiftRule
type=feat.rule.shift.semantics id=featShiftSemantics
type=feat.rule.cross id=featCrossRule
type=feat.attribute.pos id=featAttribPOS
type=feat.amr.relations id=featAttach
type=feat.lex.dyngen id=featLexDyngen
type=feat.lex.sloppy id=featLexSloppy
type=feat.lex.dynskip id=featDynSkip
type=feat.decomposed.pos id=featDecompPos
#type=feat.decomposed.syntactic.attribute id=featSyntacticAttrib
#type=feat.lex.factored id=featLexFactored
#type=feat.decomposed.amrhead id=featDecomposAmrHead
#type=feat.lookahead.pos id=featLookAheadPOS
type=feat.step.rule id=featStepRule
type=feat.template.adjacent.pos id=featTemplateAdjacentPOS
type=feat.adjacent.pos id=featAdjacentPOS
#type=feat.dyn.origin id=featDynOrigin
#type=feat.lexical.word id=featLexicalWord
#type=feat.adjacent.word id=featAdjacentWord
#type=feat.parse.step.syntax id=featParseStepSyntax
## Lexicon
type=lexicon.factored.partial id=lexicon
## Model
type=model.joint id=modelNewFeatures lexicon=lexicon features=featAttribPOS,featShiftSemantics,featLogexpCoordination,featLex,featLexDyngen,featAttach,featShiftRule,featCrossRule,featLexSloppy,featDynSkip,featStepRule,featTemplateAdjacentPOS,featDecompPos,featAdjacentPOS
#type=model.joint id=modelNewFeatures lexicon=lexicon features=featLogexpCoordination,featLexDyngen,featAttach,featShiftRule,featCrossRule,featLexSloppy,featDynSkip,featDecompPos,featSyntacticAttrib,featLexFactored,featDecomposAmrHead
