# Step 4: Translation - Requirements Document

## 1. Overview

**Step Name:** Translation

**Purpose:** Translate subtitle content from source language to target language while preserving timing information and handling special cases.

**Position in Pipeline:** Step 4 of 8 (follows Transcription, precedes Audio Clipping)

---

## 2. Input Specification

### 2.1 Primary Input
- **File:** `SRT_original.srt` (from Step 3: Transcription)
- **Format:** Standard SRT (SubRip Subtitle) format
- **Structure:**
  ```
  {sequence_number}
  {start_timestamp} --> {end_timestamp}
  {original_text}
  
  {next_sequence_number}
  {start_timestamp} --> {end_timestamp}
  {original_text}
  ```

**Example:**
```
1
00:00:01,000 --> 00:00:03,500
Hello, how are you?

2
00:00:03,600 --> 00:00:05,200
[[ laughing ]]

3
00:00:05,300 --> 00:00:08,000
I'm doing great, thank you!
```

### 2.2 Configuration Parameters

#### 2.2.1 Mandatory Parameters
- **source_language** (string): Source language of the original text
  - Provided by Step 3 (Transcription) output metadata
  - Format: Language name (e.g., "English", "Spanish") or language code (e.g., "en", "es")
  - Should be extracted from transcription metadata or passed from previous step
  - Example: `"English"`, `"Spanish"`, `"Japanese"`

- **target_language** (string): Target language for translation
  - Format: Language name (e.g., "Chinese", "Spanish", "French") or language code (e.g., "zh", "es", "fr")
  - Must be a valid language supported by the translation service
  - Example: `"Chinese"`, `"Spanish"`, `"Japanese"`

#### 2.2.2 Optional Parameters
- **terminology_reference** (JSON string or file path): Custom terminology/glossary
  - Format: JSON object as string or path to JSON file
  - Structure: `{"source_term": "target_translation"}`
  - Purpose: Ensures specific terms/phrases are translated consistently according to domain-specific requirements
  - **Implementation Note:** Can be treated as simple text and passed directly to the LLM service in the prompt
  - Example:
    ```json
    {
      "machine learning": "机器学习",
      "neural network": "神经网络",
      "API": "API"
    }
    ```

---

## 3. Translation Service Configuration

### 3.1 Service Provider
- **Provider:** OpenAI ChatGPT
- **Model:** GPT-4o-mini (chatgpt-4o-mini)
- **API:** OpenAI Chat Completions API

### 3.2 Authentication
- **Method:** API Key authentication
- **Configuration:** 
  - API key should be stored in environment variable `OPENAI_API_KEY`
  - Or provided via configuration file (not hardcoded)

### 3.3 API Parameters
- **model:** `"gpt-4o-mini"`
- **temperature:** `0.3` (low temperature for consistent, accurate translation)
- **max_tokens:** Appropriate limit based on input length
- **top_p:** `1.0`

---

## 4. Processing Logic

### 4.1 SRT Parsing
1. Parse input SRT file to extract:
   - **Sequence number** (integer): Sequential ID of each subtitle entry
   - **Start timestamp** (string): Format `HH:MM:SS,mmm` (hours:minutes:seconds,milliseconds)
   - **End timestamp** (string): Format `HH:MM:SS,mmm`
   - **Original text** (string): Content to be translated

2. Validate SRT format:
   - Ensure sequence numbers are sequential
   - Verify timestamp format is valid
   - Check for empty or malformed entries

### 4.2 Content Classification and Event Handling

Before translation, classify and process each text entry:

#### 4.2.1 Pure Non-Dialogue Events
- **Definition:** Sound effects, actions, or events indicated by double square brackets
- **Pattern:** Text matching `[[ ... ]]` pattern (entire text is enclosed in double brackets)
- **Examples:** `[[ laughing ]]`, `[[ crying ]]`, `[[ music ]]`, `[[ door slams ]]`, `[[ applause ]]`
- **Action:** **DO NOT TRANSLATE** - Keep original text as-is in output

#### 4.2.2 Mixed Content (Event + Dialogue)
- **Definition:** Text containing both event markers and dialogue
- **Pattern:** Text containing `[[ event ]]` plus additional content
- **Examples:** 
  - `[[ laughing ]] Hello there` → Extract "Hello there"
  - `I'm fine [[ coughing ]]` → Extract "I'm fine"
  - `[[ music ]] Welcome to the show [[ applause ]]` → Extract "Welcome to the show"
- **Action:** 
  1. **Remove/Ignore all event markers** `[[ ... ]]`
  2. **Extract remaining dialogue text**
  3. **Translate only the extracted dialogue**

#### 4.2.3 Pure Dialogue Text
- **Definition:** Regular spoken content without event markers
- **Characteristics:** Natural language sentences, phrases, or speech
- **Action:** Translate to target language

**Processing Logic:**
```
IF text matches pattern: ^\[\[.*\]\]$ (only event, nothing else)
  THEN: Non-dialogue event → Keep as-is
ELSE IF text contains [[ event ]] markers:
  THEN: Mixed content → Remove all [[ event ]] markers → Translate remaining text
ELSE:
  THEN: Pure dialogue → Translate entire text
```

**Implementation Note:**
- Use regex to detect and remove event patterns: `\[\[.*?\]\]`
- Trim whitespace after removing events
- If removal results in empty string, treat as pure event (don't translate)

### 4.3 Translation Prompt Construction

#### 4.3.1 Base Prompt (Without Terminology)
```
You are a professional translator. Translate the following text from {source_language} to {target_language}.

Requirements:
1. Maintain the natural tone and style of the original text
2. Preserve any formatting or special characters
3. Provide only the translated text without explanations

Text to translate: {original_text}
```

#### 4.3.2 Enhanced Prompt (With Terminology Reference)
```
You are a professional translator. Translate the following text from {source_language} to {target_language}.

Requirements:
1. Maintain the natural tone and style of the original text
2. Preserve any formatting or special characters
3. STRICTLY follow the terminology reference provided below for specific terms
4. Provide only the translated text without explanations

Terminology Reference:
{terminology_reference}

Text to translate: {original_text}
```

**Note:** The terminology reference is passed as simple text (JSON string) directly in the prompt. The LLM will interpret and apply it.

#### 4.3.3 Terminology Enforcement
- When terminology reference is provided:
  - **MUST** use the exact translations specified in the reference
  - Terms in the reference take absolute precedence over natural translation
  - Maintain consistency across all sentences
  - Case sensitivity should be preserved where applicable

### 4.4 Translation Execution

**Process Flow:**
1. For each subtitle entry in sequence:
   - Extract original text
   - Check for event markers `[...]`
   - **If pure event** (matches `^\[.*\]$`):
     - Skip translation
     - Use original text as "translation"
   - **If mixed content** (contains `[...]` with other text):
     - Remove all event markers using regex `\[.*?\]`
     - Trim whitespace from remaining text
     - If remaining text is not empty:
       - Translate the cleaned text
     - Else (empty after removal):
       - Use original text as "translation"
   - **If pure dialogue** (no event markers):
     - Construct translation prompt (with/without terminology)
     - Call ChatGPT API
     - Receive translated text
     - Validate response is not empty
2. Handle API errors with retry logic (up to 3 attempts)
3. Log translation progress and any errors

---

## 5. Output Specification

### 5.1 Primary Output - JSON Format

**File Name:** `translation_output.json`

**Format:** JSON file containing combined original and translated data

**Structure:** Array of objects, one per subtitle entry

```json
[
  {
    "seq": 1,
    "start": "00:00:01,000",
    "end": "00:00:03,500",
    "text": "Hello, how are you?",
    "translation": "你好，你好吗？"
  },
  {
    "seq": 2,
    "start": "00:00:03,600",
    "end": "00:00:05,200",
    "text": "[[ laughing ]]",
    "translation": "[[ laughing ]]"
  },
  {
    "seq": 3,
    "start": "00:00:05,300",
    "end": "00:00:08,000",
    "text": "I'm doing great, thank you!",
    "translation": "我很好，谢谢！"
  }
]
```

### 5.2 Field Specifications

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `seq` | integer | Sequence number from original SRT | Preserved from input |
| `start` | string | Start timestamp (HH:MM:SS,mmm) | Extracted from timestamp line |
| `end` | string | End timestamp (HH:MM:SS,mmm) | Extracted from timestamp line |
| `text` | string | Original text in source language | Preserved from input |
| `translation` | string | Translated text or original (for non-dialogue) | Generated or preserved |

### 5.3 Secondary Output - SRT Format

**File Name:** `SRT_translated.srt`

**Format:** Standard SRT format with translated text

**Purpose:** 
- Required for downstream steps (Step 5 and beyond)
- Allows manual review of translations
- Standard format for subtitle compatibility

**Structure:**
```
{seq}
{start} --> {end}
{translation}

{next_seq}
{start} --> {end}
{translation}
```

**Example:**
```
1
00:00:01,000 --> 00:00:03,500
你好，你好吗？

2
00:00:03,600 --> 00:00:05,200
[[ laughing ]]

3
00:00:05,300 --> 00:00:08,000
我很好，谢谢！
```

**Requirements:**
- Must be generated in addition to JSON output
- Must maintain exact SRT format specification
- Encoding: UTF-8 with BOM or without (configurable)

---

## 6. Quality Requirements

### 6.1 Translation Quality
- Maintain semantic accuracy
- Preserve emotional tone and speaking style
- Natural phrasing in target language
- Culturally appropriate expressions

### 6.2 Data Integrity
- **Sequence numbers:** Must remain unchanged and in order
- **Timestamps:** Must be preserved exactly (no modification)
- **Format:** All timestamps must remain in `HH:MM:SS,mmm --> HH:MM:SS,mmm` format
- **Completeness:** All entries from input must appear in output

### 6.3 Special Handling
- Empty or whitespace-only text: Preserve as-is
- Multiple consecutive spaces: Preserve formatting
- Punctuation: Adapt to target language conventions while preserving meaning
- Numbers and dates: Adapt to target language format if culturally appropriate

---

## 7. Error Handling

### 7.1 Input Validation Errors
- **Invalid SRT format:** Log error, identify problematic line, halt processing
- **Missing timestamps:** Log warning, skip entry or use default values
- **Malformed sequence numbers:** Attempt to reconstruct sequence, log warning

### 7.2 API Errors
- **Authentication failure:** Check API key, provide clear error message
- **Rate limiting:** Implement exponential backoff, retry up to 3 times
- **Network errors:** Retry with timeout, log failure after max attempts
- **Invalid response:** Log raw response, attempt to extract partial translation

### 7.3 Translation Errors
- **Empty translation returned:** Log warning, use original text as fallback
- **Encoding issues:** Ensure UTF-8 encoding throughout
- **Terminology mismatch:** Log if required terms not found in translation

### 7.4 Recovery Strategy
- Maintain checkpoint of successfully translated entries
- Allow resume from last successful translation
- Provide partial output if process is interrupted

---

## 8. Configuration Example

**YAML Configuration:**
```yaml
translation:
  service: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  
  # Translation parameters
  source_language: English  # From Step 3 metadata
  target_language: Chinese
  
  # Optional terminology
  terminology_reference: config/terminology.json
  
  # API settings
  temperature: 0.3
  max_tokens: 2000
  timeout: 30
  retry_attempts: 3
  retry_delay: 2  # seconds
  
  # Output settings
  output_json: true
  output_srt: true
  json_indent: 2
```

**Terminology Reference File (JSON):**
```json
{
  "video translator": "视频翻译器",
  "transcription": "转录",
  "vocal separation": "人声分离",
  "TTS": "文字转语音",
  "API": "API"
}
```

---

## 9. Implementation Guidelines

### 9.1 Modular Design
- Create abstract `Translator` base class
- Implement `OpenAITranslator` as concrete implementation
- Allow easy swapping of translation services (Google, DeepL, etc.)

### 9.2 Class Structure (Proposed)

```python
class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, target_lang: str, 
                  terminology: dict = None) -> str:
        pass

class OpenAITranslator(Translator):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        # Initialize OpenAI client
        
    def translate(self, text: str, target_lang: str,
                  terminology: dict = None) -> str:
        # Implementation
```

### 9.3 SRT Handler Utilities
- Reuse or extend existing `srt_handler.py` from `src/utils/`
- Functions needed:
  - `parse_srt(file_path) -> List[Dict]`
  - `write_srt(data, file_path)`
  - `validate_srt_format(file_path) -> bool`

---

## 10. Testing Requirements

### 10.1 Unit Tests
- SRT parsing with valid and invalid formats
- Event detection (non-dialogue classification)
- Terminology application in translations
- JSON output structure validation

### 10.2 Integration Tests
- End-to-end translation with real API
- Handling of mixed dialogue and events
- Terminology reference enforcement
- Large file processing (100+ entries)

### 10.3 Test Cases

| Test Case | Input | Expected Output | Notes |
|-----------|-------|-----------------|-------|
| Basic dialogue | "Hello world" | Translated text in target language | Standard translation |
| Pure event | "[[ laughing ]]" | "[[ laughing ]]" (unchanged) | No translation |
| Event at start | "[[ laughing ]] Hello there" | Translated "Hello there" only | Remove [[ laughing ]], translate rest |
| Event at end | "I'm fine [[ coughing ]]" | Translated "I'm fine" only | Remove [[ coughing ]], translate rest |
| Multiple events | "[[ music ]] Welcome [[ applause ]]" | Translated "Welcome" only | Remove all events |
| Event only result | "[[ music ]] [[ applause ]]" | "[[ music ]] [[ applause ]]" | After removal = empty, keep original |
| With terminology | "Use the API" + {"API": "接口"} | "使用接口" (uses term) | Strict terminology |
| Empty text | "" | "" (preserved) | Edge case |
| Whitespace | "   " | "   " (preserved) | Edge case |

---

## 11. Performance Considerations

### 11.1 Optimization
- **Batch processing:** Group multiple sentences in single API call if service supports
- **Caching:** Cache translations for repeated phrases
- **Parallel processing:** Process independent entries concurrently (respect API rate limits)

### 11.2 Rate Limiting
- Respect OpenAI API rate limits (requests per minute)
- Implement token bucket or similar algorithm
- Add configurable delay between requests if needed

### 11.3 Cost Management
- Log token usage for each API call
- Provide cost estimation before processing
- Allow user confirmation for large files

---

## 12. Success Criteria

The Step 4 implementation is successful when:

1. ✅ All dialogue entries are accurately translated to target language
2. ✅ All non-dialogue events (e.g., `[laughing]`) remain unchanged
3. ✅ Sequence numbers and timestamps are preserved exactly
4. ✅ Terminology reference terms are strictly applied when provided
5. ✅ Output JSON contains all required fields with correct data types
6. ✅ Process handles API errors gracefully with retries
7. ✅ Both JSON and SRT outputs are generated correctly
8. ✅ Translation quality is natural and contextually appropriate
9. ✅ Process completes for files with 100+ subtitle entries
10. ✅ Implementation is modular and can swap translation services

---

## 13. Dependencies

### 13.1 Python Packages
```
openai>=1.0.0          # OpenAI API client
pysrt>=1.1.2           # SRT file parsing (or custom implementation)
python-dotenv>=1.0.0   # Environment variable management
pydantic>=2.0.0        # Data validation (optional)
```

### 13.2 External Services
- OpenAI API (ChatGPT service)
- Internet connection for API calls

### 13.3 Internal Dependencies
- `src/utils/srt_handler.py` - SRT parsing and writing utilities
- `src/config.py` - Configuration management
- **Step 3 outputs:**
  - `SRT_original.srt` - Original subtitle file
  - `transcription_metadata.json` - Contains source language information

---

## 14. Future Enhancements

### 14.1 Potential Improvements
- Support for multiple translation service backends (DeepL, Google Translate, Azure)
- Automatic language detection for source language
- Translation memory for consistency across multiple videos
- Custom prompt templates for different content types
- Quality scoring and confidence metrics
- Support for context-aware translation (using previous/next sentences)
- Glossary learning from user corrections

### 14.2 Advanced Features
- A/B testing different translation models
- Human-in-the-loop review workflow
- Automatic terminology extraction from domain documents
- Style transfer (formal/informal tone adjustment)

---

## Appendix A: SRT Format Reference

**Standard SRT Structure:**
```
<sequence_number>
<start_time> --> <end_time>
<subtitle_text>
<blank line>
```

**Timestamp Format:**
- Pattern: `HH:MM:SS,mmm` (00:00:00,000 to 99:59:59,999)
- Separator: ` --> `
- Example: `00:01:23,456 --> 00:01:25,789`

**Special Characters:**
- Blank lines separate entries
- Text can span multiple lines (rare in this use case)
- UTF-8 encoding required for international characters

---

## Appendix B: Example API Call

**Python Code Example:**
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_text(text, target_lang, terminology=None):
    # Build prompt
    prompt = f"Translate to {target_lang}: {text}"
    if terminology:
        prompt = f"Using terminology {terminology}, " + prompt
    
    # Call API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-16  
**Status:** Draft for Review
