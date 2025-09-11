# Obligation Extraction from EUR-Lex Documents

## Project Overview

Fine-tune a smaller instruction-tuned model to extract legal obligations and their scope from EU regulations and directives, demonstrating improved performance over base models.

## Use Case Definition

**Task**: Extract structured obligations from EUR-Lex documents

-   **Obligations**: Legal requirements, duties, prohibitions that entities must comply with
-   **Scope**: Who/what is affected (member states, companies, individuals, sectors)
-   **Context**: When, where, under what conditions the obligation applies

## Dataset Construction Strategy

### 1. Data Collection & Preprocessing

-   **Source**: EUR-Lex corpus (regulations, directives)
-   **Document types**: Focus on high-impact areas (GDPR, environmental, financial services)
-   **Preprocessing**: Extract clean text, maintain document structure
-   **Size target**: 1000-2000 documents for initial dataset

#### Target Documents (Consolidated Versions)

**GDPR & Data Protection**

-   General Data Protection Regulation - CELEX: 02016R0679-20160504
-   ePrivacy Directive - CELEX: 02002L0058-20091219
-   EU Institutions Data Protection Regulation - CELEX: 02018R1725-20181212
-   Police Directive (Data Protection in Criminal Law) - CELEX: 02016L0680-20160504

**Environmental Law**

-   EU Deforestation Regulation - CELEX: 02023R1115-20241226
-   Industrial Emissions Directive - CELEX: 02010L0075-20240804
-   REACH Regulation (Chemicals) - CELEX: 02006R1907-20250623
-   Waste Framework Directive - CELEX: 02008L0098-20180705
-   EU Taxonomy Regulation (Sustainable Finance) - CELEX: 02020R0852-20230101

**Financial Services**

-   MiFID II (Markets in Financial Instruments) - CELEX: 02014L0065-20250117
-   Capital Requirements Regulation (CRR) - CELEX: 02013R0575-20250629
-   Capital Requirements Directive (CRD IV) - CELEX: 02013L0036-20250117
-   Payment Services Directive 2 (PSD2) - CELEX: 02015L2366-20250117
-   Benchmark Regulation - CELEX: 02016R1011-20220101

### 2. Annotation Schema

```json
{
    "document_id": "EUR-Lex identifier",
    "text_excerpt": "Original text containing obligation",
    "obligation": {
        "type": "requirement|prohibition",
        "description": "Clear summary of what must/must not be done",
        "scope_subject": "Who is obligated, who must comply with the rules",
        "scope_affected_parties": "Those who need to be aware because the obligation impacts them, even if they're not directly obligated",
        "context": "When it applies (simplified conditions)"
    }
}
```

**Rationale for Simplification**:

-   **Removed "permission"**: Focus on actionable obligations (requirements/prohibitions)
-   **Merged fields**: Combined conditions, deadlines, objects into "scope"
-   **Clearer descriptions**: Easier for LLMs to generate consistent annotations
-   **Reduced cognitive load**: AI engineers can validate 4 fields vs 7

### 3. LLM-Assisted Annotation Process

**Phase 1: LLM Bootstrap (1-2 weeks)**

-   Use GPT-4/Claude to generate initial annotations on 100-200 documents
-   AI engineers review and create "gold standard" examples
-   Develop clear annotation guidelines based on common patterns

**Phase 2: Iterative Refinement (2-3 weeks)**

-   LLM generates annotations → AI engineer validates/corrects → Update guidelines
-   Focus on edge cases and difficult examples
-   Build up to 500-800 annotated examples

**Phase 3: Quality Assurance (1 week)**

-   Cross-validation between multiple AI engineers
-   Consistency checks using agreement metrics
-   Final dataset cleaning and formatting

**Tools**:

-   **LLM API**: GPT-4 or Claude for initial annotations
-   **Validation**: Simple web interface or spreadsheet for engineer review
-   **Automation**: Scripts to batch process documents and track agreement

### LLM Annotation Workflow

```python
# Pseudo-code for annotation pipeline
def annotate_document(text, llm_client):
    prompt = f"""
    Extract legal obligations from this EUR-Lex text.
    Focus on requirements (what must be done) and prohibitions (what must not be done).

    Text: {text}

    Return JSON format:
    {{
        "obligations": [
            {{
                "type": "requirement|prohibition",
                "description": "Clear summary",
                "subject": "Who is obligated",
                "scope": "What/when it applies"
            }}
        ]
    }}
    """
    return llm_client.generate(prompt)

# Validation workflow
def validate_annotations(llm_annotations, human_reviewer):
    # Present side-by-side for engineer validation
    # Track agreement rates and common error patterns
    # Update annotation guidelines based on feedback
```

**Quality Control Metrics**:

-   **Inter-LLM agreement**: Test same document with different models
-   **Human validation rate**: % of LLM annotations accepted by engineers
-   **Consistency checks**: Ensure similar obligations are labeled consistently

### 4. Data Splitting Strategy

-   **Training**: 70% (focus on diverse obligation types)
-   **Validation**: 15% (for hyperparameter tuning)
-   **Test**: 15% (held-out for final evaluation)
-   **Stratify by**: Document type, obligation complexity, legal domain

## Model Selection & Fine-tuning

### Base Model Candidates

1. **Llama 3.1 8B Instruct** - Good balance of size/performance
2. **Mistral 7B Instruct** - Efficient, good at following instructions
3. **Phi-3.5 Mini** - Smaller, faster inference
4. **Qwen 2.5 7B Instruct** - Strong reasoning capabilities

### Fine-tuning Approach

-   **Method**: LoRA (Low-Rank Adaptation) for efficiency
-   **Format**: Instruction-following format with system prompts
-   **Training**: Supervised fine-tuning on annotated examples
-   **Validation**: Regular evaluation on held-out set

### Prompt Engineering

```
System: You are a legal AI assistant specializing in EU law. Extract obligations from the provided text.

User: Extract all legal obligations from this EUR-Lex text: [TEXT]

Format your response as JSON with the following structure for each obligation found:
{
  "obligations": [
    {
      "type": "requirement/prohibition/permission",
      "description": "...",
      "subject": "...",
      "object": "...",
      "conditions": "...",
      "deadline": "...",
      "sanctions": "..."
    }
  ]
}
```

## Evaluation Metrics & Benchmarking

### Quantitative Metrics

1. **Precision/Recall/F1**: For obligation detection
2. **Exact Match**: For complete obligation extraction
3. **BLEU/ROUGE**: For description quality
4. **Semantic Similarity**: Using embeddings for partial matches
5. **Attribute Accuracy**: Per-field evaluation (subject, object, etc.)

### Qualitative Assessment

-   **AI Engineer Review**: Manual evaluation of complex cases
-   **Error Analysis**: Categorize failure modes (missing obligations, incorrect classification, scope errors)
-   **Bias Detection**: Check for domain/document type bias

### Baseline Comparisons

1. **GPT-4/Claude-3.5**: Commercial model performance
2. **Base Model (Zero-shot)**: Pre-fine-tuning performance
3. **Rule-based System**: Traditional NLP approach
4. **Named Entity Recognition**: Simple entity extraction baseline

## Simplified Implementation Timeline (8-10 weeks total)

### Phase 1: LLM-Assisted Dataset Creation (4-5 weeks)

-   [ ] Collect and clean EUR-Lex documents (1 week)
-   [ ] Set up LLM annotation pipeline (GPT-4/Claude API) (1 week)
-   [ ] Generate initial annotations and create guidelines (1-2 weeks)
-   [ ] AI engineer validation and refinement (1-2 weeks)
-   [ ] Create train/validation/test splits (0.5 weeks)

### Phase 2: Model Development (3-4 weeks)

-   [ ] Implement baseline evaluations (0.5 weeks)
-   [ ] Set up fine-tuning pipeline with LoRA (1 week)
-   [ ] Train models on annotated dataset (1-2 weeks)
-   [ ] Hyperparameter tuning and optimization (1 week)

### Phase 3: Evaluation & Demo (1-2 weeks)

-   [ ] Comprehensive evaluation against baselines (0.5 weeks)
-   [ ] Error analysis and performance comparison (0.5 weeks)
-   [ ] Prepare demo materials and presentation (0.5-1 week)
-   [ ] Document methodology and results (0.5 weeks)

## Technical Infrastructure

### Required Tools & Libraries

-   **Annotation**: Label Studio, Prodigy, or custom React app
-   **Fine-tuning**: Transformers, PEFT (LoRA), TRL
-   **Evaluation**: Custom metrics + standard NLP libraries
-   **Data**: Datasets library, Pandas for processing

### Compute Requirements

-   **Training**: 1x A100 or 2x RTX 4090 (for 7-8B models with LoRA)
-   **Inference**: CPU/smaller GPU for demo
-   **Storage**: ~50GB for dataset and models

## Risk Mitigation

### Technical Risks

-   **Annotation Quality**: Multiple annotators, clear guidelines, quality checks
-   **Data Scarcity**: Active learning, synthetic data generation
-   **Model Performance**: Multiple base model candidates, ensemble methods

### Legal/Ethical Risks

-   **Accuracy**: Clear disclaimers about AI limitations in legal contexts
-   **Bias**: Diverse document selection, bias evaluation metrics
-   **Privacy**: Use only public EUR-Lex documents

## Success Criteria

### Demo Success

-   [ ] 20-30% improvement over base model on key metrics
-   [ ] Convincing qualitative examples of obligation extraction
-   [ ] Clear presentation of before/after model performance

### Technical Success

-   [ ] F1 score > 0.75 on obligation detection
-   [ ] Attribute accuracy > 0.65 for complex fields
-   [ ] Processing time < 10s per document for demo

## Future Extensions

### Dataset Expansion

-   **Multi-language**: Extend to other EU languages
-   **Temporal Analysis**: Track obligation changes over time
-   **Cross-reference**: Link obligations to implementing acts

### Model Improvements

-   **Multi-task Learning**: Joint extraction of obligations, definitions, exemptions
-   **Retrieval-Augmented**: Incorporate legal precedents and interpretations
-   **Explanation**: Generate rationales for extracted obligations

## Budget Estimation

### Personnel (8-10 weeks)

-   AI engineer development & validation: €10,000-12,000
-   Project management: €2,000-3,000

### Compute & Infrastructure

-   LLM API costs (annotation): €300-600
-   Cloud GPU training: €500-1,000
-   Storage and misc: €200-500

**Total Estimated Budget: €13,000-17,000**

### Cost Savings vs Expert Approach

-   **60% cost reduction** by eliminating legal expert fees
-   **30% faster timeline** with LLM-assisted annotation
-   **Scalable approach** - can easily expand dataset size
