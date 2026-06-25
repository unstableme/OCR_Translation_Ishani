import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Check, AlertCircle, RefreshCw, ArrowRight, Download, Info, Type, Camera, Mic, Square, Trash2, Volume2, VolumeX } from 'lucide-react';
import axios from 'axios';
import { notoDevanagari } from '../fonts/NotoSansDevanagari';
import { jsPDF } from 'jspdf';
import './Dashboard.css';

const API_BASE_URL = `http://${window.location.hostname}:8000`;

const TamangTranslationIcon = ({ size = 24, className = "" }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={className}
    >
        {/* Background 'page' with Devanagari 'अ' */}
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        <text x="4" y="11" fontSize="9" strokeWidth="0.5" fill="currentColor" style={{ fontFamily: 'serif', fontWeight: 'bold' }}>अ</text>

        {/* Foreground 'page' with Latin 'A' */}
        <path d="m22 22-5-10-5 10" />
        <path d="M14 18h6" />
        <path d="M12 11h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-9a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2z" fill="none" />
        <text x="14" y="21" fontSize="10" strokeWidth="0.5" fill="currentColor" style={{ fontFamily: 'Arial', fontWeight: 'bold' }}>A</text>
    </svg>
);

const LOADING_MESSAGES = [
    "Initializing AI processing...",
    "Scanning document for text...",
    "Analyzing layout and structure...",
    "Running OCR extraction...",
    "Translating your document...",
    "Optimizing response for readability...",
    "Finalizing your results..."
];

const RANJANA_BASE_MAP = {
    "अ": "c", "आ": "cf", "इ": "O", "ई": "O{", "उ": "p", "ऊ": "pm", "ऋ": "C",
    "ए": "P", "ऐ": "P]", "ओ": "cf]", "औ": "cf}", "क": "s", "ख": "v", "ग": "u",
    "घ": "3", "ङ": "ª", "च": "r", "छ": "5", "ज": "h", "झ": "em", "ञ": "`",
    "ट": "6", "ठ": "7", "ड": "8", "ढ": "9", "ण": "0f", "त": "t", "थ": "y",
    "द": "b", "ध": "w", "न": "g", "प": "k", "फ": "km", "ब": "a", "भ": "e",
    "म": "d", "य": "o", "र": "/", "ल": "n", "व": "j", "श": "z", "ष": "if",
    "स": ";", "ह": "x", "क्ष": "If", "त्र": "q", "ज्ञ": "1", "श्र": ">", "।": ".",
    "०": ")", "१": "!", "२": "@", "३": "#", "४": "$", "५": "%", "६": "^",
    "७": "&", "८": "*", "९": "(",
};

const RANJANA_MARK_MAP = {
    "ा": "f", "ी": "L", "ु": "'", "ू": "\"", "ृ": "[", "े": "]", "ै": "}",
    "ो": "f]", "ौ": "f}", "ं": "+", "ः": "M", "ँ": "F", "्": "",
};

const RANJANA_SPECIAL_CLUSTERS = ["क्ष", "त्र", "ज्ञ", "श्र"];
const RANJANA_BASE_FROM_LEGACY = Object.entries(RANJANA_BASE_MAP)
    .sort(([, a], [, b]) => b.length - a.length)
    .map(([devanagari, legacy]) => ({ legacy, devanagari }));
const RANJANA_MARK_FROM_LEGACY = Object.entries(RANJANA_MARK_MAP)
    .filter(([, legacy]) => legacy)
    .sort(([, a], [, b]) => b.length - a.length)
    .map(([devanagari, legacy]) => ({ legacy, devanagari }));
const DEVANAGARI_PATTERN = /[\u0900-\u097F]/;

const TAMYIG_BASE_MAP = {
    "अ": "ཨ", "आ": "ཨཱ", "इ": "ཨི", "ई": "ཨཱི", "उ": "ཨུ", "ऊ": "ཨཱུ",
    "ए": "ཨེ", "ऐ": "ཨཻ", "ओ": "ཨོ", "औ": "ཨཽ",
    "क": "ཀ", "ख": "ཁ", "ग": "ག", "घ": "གྷ", "ङ": "ང",
    "च": "ཅ", "छ": "ཆ", "ज": "ཇ", "झ": "ཇྷ", "ञ": "ཉ",
    "ट": "ཊ", "ठ": "ཋ", "ड": "ཌ", "ढ": "ཌྷ", "ण": "ཎ",
    "त": "ཏ", "थ": "ཐ", "द": "ད", "ध": "དྷ", "न": "ན",
    "प": "པ", "फ": "ཕ", "ब": "བ", "भ": "བྷ", "म": "མ",
    "य": "ཡ", "र": "ར", "ल": "ལ", "व": "ཝ",
    "श": "ཤ", "ष": "ཥ", "स": "ས", "ह": "ཧ",
    "क़": "ཀ", "ख़": "ཁ", "ग़": "ག", "ज़": "ཛ", "ड़": "ཌ", "ढ़": "ཌྷ", "फ़": "ཕ",
};

const TAMYIG_SUBJOINED_MAP = {
    "क": "ྐ", "ख": "ྑ", "ग": "ྒ", "ङ": "ྔ",
    "च": "ྕ", "छ": "ྖ", "ज": "ྗ", "ञ": "ྙ",
    "ट": "ྚ", "ठ": "ྛ", "ड": "ྜ", "ण": "ྞ",
    "त": "ྟ", "थ": "ྠ", "द": "ྡ", "न": "ྣ",
    "प": "ྤ", "फ": "ྥ", "ब": "ྦ", "म": "ྨ",
    "य": "ྱ", "र": "ྲ", "ल": "ླ",
    "श": "ྴ", "ष": "ྵ", "स": "ྶ", "ह": "ྷ",
};

const TAMYIG_MARK_MAP = {
    "ा": "ཱ", "ि": "ི", "ी": "ཱི", "ु": "ུ", "ू": "ཱུ", "ृ": "ྲྀ",
    "े": "ེ", "ै": "ཻ", "ो": "ོ", "ौ": "ཽ", "ं": "ཾ", "ः": "ཿ", "ँ": "ྃ",
};

const TAMYIG_DIGIT_MAP = {
    "०": "༠", "१": "༡", "२": "༢", "३": "༣", "४": "༤",
    "५": "༥", "६": "༦", "७": "༧", "८": "༨", "९": "༩",
};

const TAMYIG_PUNCTUATION_MAP = {
    "।": "།", "॥": "༎",
};

const TAMYIG_BASE_FROM_UNICODE = Object.entries(TAMYIG_BASE_MAP)
    .sort(([, a], [, b]) => b.length - a.length)
    .map(([devanagari, tamyig]) => ({ tamyig, devanagari }));
const TAMYIG_SUBJOINED_FROM_UNICODE = Object.entries(TAMYIG_SUBJOINED_MAP)
    .sort(([, a], [, b]) => b.length - a.length)
    .map(([devanagari, tamyig]) => ({ tamyig, devanagari }));
const TAMYIG_MARK_FROM_UNICODE = Object.entries(TAMYIG_MARK_MAP)
    .sort(([, a], [, b]) => b.length - a.length)
    .map(([devanagari, tamyig]) => ({ tamyig, devanagari }));
const TAMYIG_DIGIT_FROM_UNICODE = Object.fromEntries(
    Object.entries(TAMYIG_DIGIT_MAP).map(([devanagari, tamyig]) => [tamyig, devanagari])
);
const TAMYIG_PUNCTUATION_FROM_UNICODE = Object.fromEntries(
    Object.entries(TAMYIG_PUNCTUATION_MAP).map(([devanagari, tamyig]) => [tamyig, devanagari])
);

const findTamyigToken = (tokens, text, index) => (
    tokens.find(({ tamyig }) => text.startsWith(tamyig, index))
);

const convertToTamyigUnicode = (value) => {
    const text = Array.isArray(value) ? value.join("\n\n") : String(value || "");
    let output = "";

    for (let i = 0; i < text.length; i += 1) {
        const twoCharToken = text.slice(i, i + 2);
        if (TAMYIG_PUNCTUATION_MAP[twoCharToken]) {
            output += TAMYIG_PUNCTUATION_MAP[twoCharToken];
            i += 1;
            continue;
        }

        const char = text[i];
        if (TAMYIG_DIGIT_MAP[char]) {
            output += TAMYIG_DIGIT_MAP[char];
            continue;
        }
        if (TAMYIG_PUNCTUATION_MAP[char]) {
            output += TAMYIG_PUNCTUATION_MAP[char];
            continue;
        }

        const base = TAMYIG_BASE_MAP[char];
        if (!base) {
            output += TAMYIG_MARK_MAP[char] ?? char;
            continue;
        }

        output += base;

        while (text[i + 1] === "्" && TAMYIG_SUBJOINED_MAP[text[i + 2]]) {
            output += TAMYIG_SUBJOINED_MAP[text[i + 2]];
            i += 2;
        }

        if (TAMYIG_MARK_MAP[text[i + 1]]) {
            output += TAMYIG_MARK_MAP[text[i + 1]];
            i += 1;
        }
    }

    return output;
};

const convertFromTamyigUnicode = (value) => {
    const text = Array.isArray(value) ? value.join("\n\n") : String(value || "");
    if (DEVANAGARI_PATTERN.test(text)) return text;

    let output = "";

    for (let i = 0; i < text.length;) {
        const twoCharToken = text.slice(i, i + 2);
        if (TAMYIG_PUNCTUATION_FROM_UNICODE[twoCharToken]) {
            output += TAMYIG_PUNCTUATION_FROM_UNICODE[twoCharToken];
            i += 2;
            continue;
        }

        const char = text[i];
        if (TAMYIG_DIGIT_FROM_UNICODE[char]) {
            output += TAMYIG_DIGIT_FROM_UNICODE[char];
            i += 1;
            continue;
        }
        if (TAMYIG_PUNCTUATION_FROM_UNICODE[char]) {
            output += TAMYIG_PUNCTUATION_FROM_UNICODE[char];
            i += 1;
            continue;
        }

        const baseToken = findTamyigToken(TAMYIG_BASE_FROM_UNICODE, text, i);
        if (baseToken) {
            output += baseToken.devanagari;
            i += baseToken.tamyig.length;

            while (i < text.length) {
                const subjoinedToken = findTamyigToken(TAMYIG_SUBJOINED_FROM_UNICODE, text, i);
                if (!subjoinedToken) break;
                output += `्${subjoinedToken.devanagari}`;
                i += subjoinedToken.tamyig.length;
            }

            const markToken = findTamyigToken(TAMYIG_MARK_FROM_UNICODE, text, i);
            if (markToken) {
                output += markToken.devanagari;
                i += markToken.tamyig.length;
            }
            continue;
        }

        const markToken = findTamyigToken(TAMYIG_MARK_FROM_UNICODE, text, i);
        if (markToken) {
            output += markToken.devanagari;
            i += markToken.tamyig.length;
            continue;
        }

        output += char;
        i += 1;
    }

    return output;
};

const convertToRanjanaLegacy = (value) => {
    const text = Array.isArray(value) ? value.join("\n\n") : String(value || "");
    let output = "";

    for (let i = 0; i < text.length; i += 1) {
        const twoCharCluster = text.slice(i, i + 2);
        const threeCharCluster = text.slice(i, i + 3);
        const cluster = RANJANA_SPECIAL_CLUSTERS.find((item) => (
            threeCharCluster === item || twoCharCluster === item
        ));

        if (cluster) {
            const next = text[i + cluster.length];
            output += next === "ि"
                ? `l${RANJANA_BASE_MAP[cluster]}`
                : RANJANA_BASE_MAP[cluster];
            if (next === "ि") i += 1;
            i += cluster.length - 1;
            continue;
        }

        const char = text[i];
        const next = text[i + 1];

        if (RANJANA_BASE_MAP[char]) {
            output += next === "ि" ? `l${RANJANA_BASE_MAP[char]}` : RANJANA_BASE_MAP[char];
            if (next === "ि") i += 1;
            continue;
        }

        output += RANJANA_MARK_MAP[char] ?? char;
    }

    return output;
};

const findRanjanaToken = (tokens, text, index) => (
    tokens.find(({ legacy }) => text.startsWith(legacy, index))
);

const convertFromRanjanaLegacy = (value) => {
    const text = Array.isArray(value) ? value.join("\n\n") : String(value || "");
    if (DEVANAGARI_PATTERN.test(text)) return text;

    let output = "";
    let pendingShortI = false;

    for (let i = 0; i < text.length;) {
        if (text[i] === "l") {
            pendingShortI = true;
            i += 1;
            continue;
        }

        const baseToken = findRanjanaToken(RANJANA_BASE_FROM_LEGACY, text, i);
        if (baseToken) {
            output += baseToken.devanagari;
            i += baseToken.legacy.length;

            if (pendingShortI) {
                output += "ि";
                pendingShortI = false;
            }

            while (i < text.length) {
                const markToken = findRanjanaToken(RANJANA_MARK_FROM_LEGACY, text, i);
                if (!markToken) break;
                output += markToken.devanagari;
                i += markToken.legacy.length;
            }
            continue;
        }

        if (pendingShortI) {
            output += "ि";
            pendingShortI = false;
        }

        output += text[i];
        i += 1;
    }

    if (pendingShortI) output += "ि";
    return output;
};



const LoadingState = () => {
    const [msgIndex, setMsgIndex] = React.useState(0);

    React.useEffect(() => {
        const interval = setInterval(() => {
            setMsgIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
        }, 4000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="loading-state-container">
            <div className="ai-scanning-animation">
                {[...Array(5)].map((_, i) => (
                    <div key={i} className="scan-line" style={{ animationDelay: `${i * 0.5}s` }}></div>
                ))}
                <TamangTranslationIcon size={80} className="floating-icon" />
            </div>
            <div className="loading-status-area">
                <div className="status-text-wrapper">
                    <p className="status-message animate-slide-up">{LOADING_MESSAGES[msgIndex]}</p>
                </div>
                <div className="progress-bar-container">
                    <div className="progress-bar-fill"></div>
                </div>
                <p className="loading-hint">This usually takes about 5-10 seconds</p>
            </div>
        </div>
    );
};

const TypewriterText = ({ text = "", speed = 30 }) => {
    const [displayedText, setDisplayedText] = useState("");
    const [index, setIndex] = useState(0);

    // Ensure we are working with a string
    const safeText = React.useMemo(() => {
        if (Array.isArray(text)) return text.join("\n\n");
        if (typeof text !== "string") return String(text || "");
        return text;
    }, [text]);

    React.useEffect(() => {
        setDisplayedText("");
        setIndex(0);
    }, [safeText]);

    React.useEffect(() => {
        const words = safeText.split(" ");
        if (index < words.length && safeText) {
            const timer = setTimeout(() => {
                setDisplayedText((prev) => prev + (prev ? " " : "") + (words[index] || ""));
                setIndex(index + 1);
            }, speed);
            return () => clearTimeout(timer);
        }
    }, [index, safeText, speed]);

    return (
        <div className="typewriter-content">
            {displayedText}
            {safeText && index < safeText.split(" ").length && (
                <span className="typewriter-cursor">|</span>
            )}
        </div>
    );
};

const Dashboard = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('translated'); // 'extracted' | 'translated'
    const [sourceLang, setSourceLang] = useState('Tamang/Newari');
    const [targetLang, setTargetLang] = useState('Nepali');
    const [inputMode, setInputMode] = useState('file'); // 'file' | 'text' | 'ranjana' | 'tamyig' | 'audio'
    const [inputText, setInputText] = useState('');
    const [cameraMode, setCameraMode] = useState(false);
    const [useRanjanaFont, setUseRanjanaFont] = useState(false);
    const [useTamyigFont, setUseTamyigFont] = useState(false);
    const [ranjanaPreviewText, setRanjanaPreviewText] = useState('');
    const [ranjanaPreviewReady, setRanjanaPreviewReady] = useState(false);
    const [ranjanaActionMsg, setRanjanaActionMsg] = useState('');
    const [tamyigPreviewText, setTamyigPreviewText] = useState('');
    const [tamyigPreviewReady, setTamyigPreviewReady] = useState(false);
    const [tamyigActionMsg, setTamyigActionMsg] = useState('');
    const [ocrDraftText, setOcrDraftText] = useState('');

    // Speech Synthesis State
    const [isReading, setIsReading] = useState(false);

    // Audio State
    const [isRecording, setIsRecording] = useState(false);
    const [, setAudioBlob] = useState(null);
    const [recordingTime, setRecordingTime] = useState(0);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const timerRef = useRef(null);

    // Live Transcription State
    const wsRef = useRef(null);
    const [transcript, setTranscript] = useState('');       // live + final transcript
    const [copied, setCopied] = useState(false);
    const [selectedEngine, setSelectedEngine] = useState('auto'); // 'auto' | 'native' | 'groq/whisper-large-v3' | 'groq/whisper-large-v3-turbo' | 'deepgram' | 'local'
    const [liveStatus, setLiveStatus] = useState('idle');   // idle | connecting | live | processing | done | error
    const [wsStatusMsg, setWsStatusMsg] = useState('');
    const transcriptRef = useRef('');                       // mirror of transcript for WS callbacks
    const liveViewRef = useRef(null);
    const [transcriptionModels, setTranscriptionModels] = useState([]);

    useEffect(() => {
        const fetchModels = async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/transcription-models`);
                if (response.data && response.data.data) {
                    const apiModels = response.data.data.map((model) => ({
                        ...model,
                        modelName: model.modelName || model.value,
                    }));
                    const browserModel = { id: 'browser-speech-API', modelName: 'browser-speech-API' };
                    setTranscriptionModels([browserModel, ...apiModels]);
                }
            } catch (err) {
                console.error("Failed to load transcription models:", err);
            }
        };
        fetchModels();
    }, []);

    const [, setActiveProvider] = useState(''); // e.g., 'Chrome Native', 'Groq Whisper'
    const [nativeSpeechLang, setNativeSpeechLang] = useState('ne-NP'); // 'ne-NP' | 'en-US' | 'hi-IN'
    const recognitionRef = useRef(null);

    useEffect(() => {
        if (!sourceLang) return;
        const lowerSrc = sourceLang.toLowerCase();
        if (lowerSrc === 'english') {
            setNativeSpeechLang('en-US');
        } else if (lowerSrc === 'hindi') {
            setNativeSpeechLang('hi-IN');
        } else {
            // Tamang/Newari/Nepali/etc default to Nepali representation
            setNativeSpeechLang('ne-NP');
        }
    }, [sourceLang]);
    const isNativeRef = useRef(false);
    const isRecordingRef = useRef(false);
    const fallbackCalledRef = useRef(false);

    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const resultsRef = useRef(null);

    const translatedDisplayText = React.useMemo(() => {
        if (!result?.translated_text) return "";
        if (useRanjanaFont) return convertToRanjanaLegacy(result.translated_text);
        if (useTamyigFont) return convertToTamyigUnicode(result.translated_text);
        return result.translated_text;
    }, [result?.translated_text, useRanjanaFont, useTamyigFont]);
    const isOcrReview = result?.workflow_stage === 'ocr_review';

    const generateRanjanaPreview = () => {
        const normalizedText = convertFromRanjanaLegacy(inputText);
        setRanjanaPreviewText(normalizedText);
        setRanjanaPreviewReady(true);
    };

    const handleRanjanaInputChange = (value) => {
        setInputText(value);
        setRanjanaPreviewReady(false);
        setRanjanaPreviewText('');
        setRanjanaActionMsg('');
        setTamyigPreviewReady(false);
        setTamyigPreviewText('');
        setTamyigActionMsg('');
    };

    const encodeDevanagariInputAsRanjana = () => {
        const encodedText = convertToRanjanaLegacy(inputText);
        setInputText(encodedText);
        setRanjanaPreviewText('');
        setRanjanaPreviewReady(false);
        setRanjanaActionMsg('Encoded for the Ranjana font.');
    };

    const copyRanjanaEncodedText = async () => {
        const encodedText = DEVANAGARI_PATTERN.test(inputText)
            ? convertToRanjanaLegacy(inputText)
            : inputText;
        await navigator.clipboard.writeText(encodedText);
        setRanjanaActionMsg('Ranjana encoded text copied.');
    };

    const generateTamyigPreview = () => {
        const normalizedText = convertFromTamyigUnicode(inputText);
        setTamyigPreviewText(normalizedText);
        setTamyigPreviewReady(true);
    };

    const handleTamyigInputChange = (value) => {
        setInputText(value);
        setRanjanaPreviewReady(false);
        setRanjanaPreviewText('');
        setRanjanaActionMsg('');
        setTamyigPreviewReady(false);
        setTamyigPreviewText('');
        setTamyigActionMsg('');
    };

    const encodeDevanagariInputAsTamyig = () => {
        const encodedText = convertToTamyigUnicode(inputText);
        setInputText(encodedText);
        setTamyigPreviewText('');
        setTamyigPreviewReady(false);
        setTamyigActionMsg('Encoded for the Tamyig font.');
    };

    const copyTamyigEncodedText = async () => {
        const encodedText = DEVANAGARI_PATTERN.test(inputText)
            ? convertToTamyigUnicode(inputText)
            : inputText;
        await navigator.clipboard.writeText(encodedText);
        setTamyigActionMsg('Tamyig encoded text copied.');
    };

    useEffect(() => {
        return () => {
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            if (recognitionRef.current) {
                recognitionRef.current.abort();
            }
        };
    }, []);

    const handleReadAloud = () => {
        if (!result) return;

        if (isReading) {
            window.speechSynthesis.cancel();
            setIsReading(false);
            return;
        }

        const text = activeTab === 'translated'
            ? result.translated_text
            : isOcrReview
                ? ocrDraftText
                : result.extracted_text;

        if (!text) return;

        const utterance = new SpeechSynthesisUtterance(text);

        // Try to set Nepali/Hindi voice for Devanagari script
        const voices = window.speechSynthesis.getVoices();
        const nepaliVoice = voices.find(voice => voice.lang.includes('ne') || voice.lang.includes('NE'));
        if (nepaliVoice) {
            utterance.voice = nepaliVoice;
        } else {
            const hindiVoice = voices.find(voice => voice.lang.includes('hi') || voice.lang.includes('HI'));
            if (hindiVoice) {
                utterance.voice = hindiVoice;
            }
        }

        utterance.lang = activeTab === 'translated' && targetLang === 'Nepali' ? 'ne-NP' : 'hi-IN';

        utterance.onstart = () => setIsReading(true);
        utterance.onend = () => setIsReading(false);
        utterance.onerror = () => setIsReading(false);

        window.speechSynthesis.speak(utterance);
    };

    const startDesktopCamera = async () => {
        setError(null);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1920 }, height: { ideal: 1080 } }
            });
            setCameraMode(true);
            setTimeout(() => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    streamRef.current = stream;
                }
            }, 100);
        } catch (err) {
            console.error("Camera error:", err);
            setError("Could not access webcam. Please ensure you have granted camera permissions.");
        }
    };

    const getBrowserNativeName = () => {
        const userAgent = navigator.userAgent;
        if (userAgent.indexOf("Edg") > -1) {
            return "Edge Native Speech";
        } else if (userAgent.indexOf("Chrome") > -1) {
            return "Chrome Native Speech";
        } else if (userAgent.indexOf("Safari") > -1 && userAgent.indexOf("Chrome") === -1) {
            return "Safari Native Speech";
        }
        return "Browser Native Speech";
    };

    const getCleanProviderName = (modelStr) => {
        if (!modelStr) return 'Cloud AI';
        if (modelStr === 'browser-speech-API') return 'Browser-Native Speech API';
        if (modelStr.includes('groq')) {
            const part = modelStr.split('/').pop();
            return `Groq Whisper (${part})`;
        }
        if (modelStr.includes('deepgram')) {
            return 'Deepgram Whisper';
        }
        if (modelStr.includes('local')) {
            return 'Local Offline Whisper';
        }
        return modelStr;
    };

    const getModelOptionValue = (model) => {
        if (model.modelName === 'browser-speech-API') return model.modelName;
        return String(model.id ?? model.modelName ?? model.value);
    };

    const getSelectedModelName = () => {
        const selectedModel = transcriptionModels.find((model) => (
            String(model.id) === String(selectedEngine) ||
            model.modelName === selectedEngine ||
            model.value === selectedEngine
        ));
        return selectedModel?.modelName || selectedModel?.value || selectedEngine;
    };

    const fallbackToWebSocket = async () => {
        if (fallbackCalledRef.current) return;
        fallbackCalledRef.current = true;
        isNativeRef.current = false;

        console.log("SpeechRecognition fallback triggered: switching to WebSocket...");

        if (recognitionRef.current) {
            try {
                recognitionRef.current.abort();
            } catch (e) {
                console.error("Error aborting native speech recognition:", e);
            }
            recognitionRef.current = null;
        }

        setLiveStatus('connecting');
        setWsStatusMsg('Switching to Cloud AI...');
        setActiveProvider('Groq Whisper (Cloud Fallback)');

        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.hostname}:8000/ws/transcribe?lang=${encodeURIComponent(sourceLang)}`;
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                setLiveStatus('live');
                setWsStatusMsg('Listening (Cloud Fallback)...');

                // Immediately send all pre-recorded audio chunks accumulated in background MediaRecorder
                if (audioChunksRef.current && audioChunksRef.current.length > 0) {
                    audioChunksRef.current.forEach((chunk) => {
                        if (chunk.size > 100) {
                            chunk.arrayBuffer().then((buf) => {
                                if (ws.readyState === WebSocket.OPEN) {
                                    ws.send(buf);
                                }
                            }).catch((err) => {
                                console.error("Error processing chunk buffer:", err);
                            });
                        }
                    });
                }
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'segment' && msg.text) {
                        transcriptRef.current = msg.text;
                        setTranscript(msg.text);
                        setWsStatusMsg('Listening (Cloud Fallback)...');
                        if (msg.model_used) {
                            setActiveProvider(getCleanProviderName(msg.model_used));
                        }
                        setTimeout(() => {
                            if (liveViewRef.current) {
                                liveViewRef.current.scrollTop = liveViewRef.current.scrollHeight;
                            }
                        }, 50);
                    } else if (msg.type === 'status') {
                        setWsStatusMsg(msg.message || '');
                    } else if (msg.type === 'done') {
                        setLiveStatus('done');
                        setWsStatusMsg('Transcription complete.');
                    } else if (msg.type === 'error') {
                        setError(`Transcription Fallback Error: ${msg.message}`);
                        setLiveStatus('error');
                    }
                } catch (err) {
                    console.warn("Failed to parse transcription message:", err);
                }
            };

            ws.onerror = () => {
                setError('WebSocket error. Falling back to local offline Whisper on stop.');
                setLiveStatus('live');
                setWsStatusMsg('Recording locally...');
                setActiveProvider('Local Recording');
            };

            ws.onclose = () => {
                // Keep current done state or idle if appropriate
            };

        } catch (err) {
            console.error('WebSocket connection error:', err);
            setError('Could not connect to Cloud transcription. Recording audio locally...');
            setLiveStatus('live');
            setWsStatusMsg('Recording locally...');
            setActiveProvider('Local Recording');
        }
    };

    // Audio Recording Logic — Browser Native Speech Recognition with Progressive Fallbacks
    const startRecording = async () => {
        setError(null);
        setTranscript('');
        transcriptRef.current = '';
        setLiveStatus('connecting');
        setWsStatusMsg('Initializing...');

        isRecordingRef.current = true;
        fallbackCalledRef.current = false;

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const useNative = selectedEngine === 'browser-speech-API' || (selectedEngine === 'auto' && SpeechRecognition);

        if (useNative && SpeechRecognition) {
            // Try Browser Native Speech Recognition
            isNativeRef.current = true;
            setActiveProvider(getBrowserNativeName());
            setLiveStatus('live');
            setWsStatusMsg('Listening...');
            setIsRecording(true);

            // Update timer
            setRecordingTime(0);
            timerRef.current = setInterval(() => {
                setRecordingTime((prev) => prev + 1);
            }, 1000);

            const rec = new SpeechRecognition();
            rec.continuous = true;
            rec.interimResults = true;

            rec.lang = nativeSpeechLang;

            let finalTranscript = '';
            rec.onresult = (event) => {
                let interimTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript + ' ';
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                const currentText = (finalTranscript + interimTranscript).trim();
                if (currentText) {
                    setTranscript(currentText);
                    transcriptRef.current = currentText;
                    setTimeout(() => {
                        if (liveViewRef.current) {
                            liveViewRef.current.scrollTop = liveViewRef.current.scrollHeight;
                        }
                    }, 50);
                }
            };

            rec.onerror = (event) => {
                console.error("SpeechRecognition error:", event.error);
                if (event.error === 'not-allowed') {
                    setError("Microphone permission denied.");
                    stopRecording();
                    return;
                }
                if (event.error === 'no-speech' || event.error === 'aborted') {
                    return;
                }
                // If the engine is browser-speech-API, do not fall back to cloud!
                if (selectedEngine === 'browser-speech-API') {
                    setError(`Speech recognition error: ${event.error}`);
                    stopRecording();
                    return;
                }
                // For other critical errors, fall back to backend WebSocket
                if (isNativeRef.current) {
                    fallbackToWebSocket();
                }
            };

            rec.onend = () => {
                // Restart only if we are still recording and native is still the active driver
                if (isRecordingRef.current && isNativeRef.current) {
                    try {
                        rec.start();
                    } catch (e) {
                        console.warn("Failed to restart speech recognition:", e);
                    }
                }
            };

            recognitionRef.current = rec;
            try {
                rec.start();
            } catch (e) {
                console.error("Error starting SpeechRecognition:", e);
                setError("Failed to start speech recognition.");
                setLiveStatus('error');
                isRecordingRef.current = false;
                setIsRecording(false);
                clearInterval(timerRef.current);
            }

        } else {
            // Check secure context and mediaDevices support
            if (!window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
                setError('Microphone access requires a secure connection (HTTPS).');
                setLiveStatus('error');
                isRecordingRef.current = false;
                return;
            }
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                setError('Your browser does not support audio recording.');
                setLiveStatus('error');
                isRecordingRef.current = false;
                return;
            }

            try {
                // Browser SpeechRecognition NOT supported/requested or fallback -> Immediate WebSocket connection
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

                const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : 'audio/webm';

                const mediaRecorder = new MediaRecorder(stream, { mimeType });
                mediaRecorderRef.current = mediaRecorder;
                audioChunksRef.current = [];

                // Update timer
                setRecordingTime(0);
                timerRef.current = setInterval(() => {
                    setRecordingTime((prev) => prev + 1);
                }, 1000);

                setIsRecording(true);
                isNativeRef.current = false;

                // Initialize WebSocket immediately
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                let wsUrl = `${wsProtocol}//${window.location.hostname}:8000/ws/transcribe?lang=${encodeURIComponent(sourceLang)}`;
                if (selectedEngine !== 'auto' && selectedEngine !== 'browser-speech-API') {
                    wsUrl += `&model=${encodeURIComponent(selectedEngine)}`;
                }
                const ws = new WebSocket(wsUrl);
                wsRef.current = ws;

                const selectedModelName = getSelectedModelName();
                const providerName = selectedEngine === 'browser-speech-API'
                    ? 'Browser Native Speech (Cloud Fallback)'
                    : getCleanProviderName(selectedModelName);
                setActiveProvider(providerName);

                ws.onopen = () => {
                    setLiveStatus('live');
                    setWsStatusMsg('Listening...');
                };

                ws.onmessage = (event) => {
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'segment' && msg.text) {
                            transcriptRef.current = msg.text;
                            setTranscript(msg.text);
                            setWsStatusMsg('Listening...');
                            if (msg.model_used) {
                                setActiveProvider(getCleanProviderName(msg.model_used));
                            }
                            setTimeout(() => {
                                if (liveViewRef.current) {
                                    liveViewRef.current.scrollTop = liveViewRef.current.scrollHeight;
                                }
                            }, 50);
                        } else if (msg.type === 'status') {
                            setWsStatusMsg(msg.message || '');
                        } else if (msg.type === 'done') {
                            setLiveStatus('done');
                            setWsStatusMsg('Transcription complete.');
                        } else if (msg.type === 'error') {
                            setError(`Transcription: ${msg.message}`);
                            setLiveStatus('error');
                        }
                    } catch (err) {
                        console.warn("Failed to parse transcription message:", err);
                    }
                };

                ws.onerror = () => {
                    setError('WebSocket connection error. Check backend status.');
                    setLiveStatus('error');
                };

                ws.onclose = () => {
                    if (liveStatus !== 'done') setLiveStatus('idle');
                };

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 100 && ws.readyState === WebSocket.OPEN) {
                        audioChunksRef.current.push(event.data);
                        event.data.arrayBuffer().then((buf) => {
                            if (ws.readyState === WebSocket.OPEN) {
                                ws.send(buf);
                            }
                        }).catch(() => { });
                    }
                };

                mediaRecorder.onstop = () => {
                    const blob = new Blob(audioChunksRef.current, { type: mimeType });
                    setAudioBlob(blob);
                    stream.getTracks().forEach((t) => t.stop());
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send('done');
                    }
                };

                mediaRecorder.start(3000);

            } catch (err) {
                console.error('Audio recording error:', err);
                setError('Could not access microphone. Please grant microphone permissions.');
                setLiveStatus('error');
                isRecordingRef.current = false;
            }
        }
    };

    const stopRecording = () => {
        isRecordingRef.current = false;

        // Stop native speech recognition if active
        if (isNativeRef.current && recognitionRef.current) {
            isNativeRef.current = false;
            try {
                recognitionRef.current.stop();
            } catch (e) {
                console.error("Error stopping native recognition:", e);
            }
        }

        // Stop media recorder
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                mediaRecorderRef.current.stop();
            } catch (e) {
                console.error("Error stopping media recorder:", e);
            }
        }

        setIsRecording(false);
        clearInterval(timerRef.current);
        setLiveStatus('done');
    };

    const clearAudio = () => {
        isRecordingRef.current = false;
        isNativeRef.current = false;
        fallbackCalledRef.current = false;

        // Abort native speech recognition
        if (recognitionRef.current) {
            try {
                recognitionRef.current.abort();
            } catch (err) {
                console.warn("Error aborting speech recognition:", err);
            }
            recognitionRef.current = null;
        }

        // Stop recording if active
        if (isRecording) {
            stopRecording();
        }

        // Close WS if open
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.close();
        }
        wsRef.current = null;
        setIsRecording(false);
        setAudioBlob(null);
        setFile(null);
        setRecordingTime(0);
        setTranscript('');
        transcriptRef.current = '';
        setLiveStatus('idle');
        setWsStatusMsg('');
        setActiveProvider('');
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    const handleCopyTranscript = () => {
        navigator.clipboard.writeText(transcript);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const stopDesktopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        setCameraMode(false);
    };

    const captureFromDesktop = () => {
        const video = videoRef.current;
        if (video) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            canvas.toBlob((blob) => {
                const capturedFile = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
                handleFileChange({ target: { files: [capturedFile] } });
                stopDesktopCamera();
            }, 'image/jpeg', 0.95);
        }
    };

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setResult(null);
            setOcrDraftText('');
            setError(null);
            setUseRanjanaFont(false);
            setUseTamyigFont(false);

            // Preview logic
            if (selectedFile.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onloadend = () => {
                    setPreviewUrl(reader.result);
                };
                reader.readAsDataURL(selectedFile);
            } else if (selectedFile.type.startsWith('audio/')) {
                setPreviewUrl(null);
                setAudioBlob(selectedFile);
                setInputMode('audio');
            } else {
                setPreviewUrl(null);
            }
        }
    };

    const handleUseSample = async (sampleName = 'Tamang_Nep.pdf') => {
        setLoading(true);
        setError(null);
        try {
            // Fetch the requested sample from the public folder
            const response = await fetch(`/${sampleName}`);
            if (!response.ok) throw new Error('Failed to load sample document');

            const blob = await response.blob();
            const sampleFile = new File([blob], sampleName, { type: 'application/pdf' });

            setFile(sampleFile);
            setPreviewUrl(null);
            setResult(null);
            setOcrDraftText('');
            setUseRanjanaFont(false);
            setUseTamyigFont(false);
        } catch (err) {
            console.error(err);
            setError(`Could not load ${sampleName} document.`);
        } finally {
            setLoading(false);
        }
    };

    const handleTranslate = async () => {
        const isFileReviewStep = inputMode === 'file' && result?.workflow_stage === 'ocr_review';

        // Audio mode: translate the live transcript text
        if (inputMode === 'audio') {
            if (!transcript.trim()) {
                setError('Please record audio first to get a transcript, then click Translate.');
                return;
            }
            setLoading(true);
            setError(null);
            setResult(null);
            setUseRanjanaFont(false);
            setUseTamyigFont(false);
            try {
                const response = await axios.post(`${API_BASE_URL}/translate`, {
                    text: transcript,
                    source_lang: sourceLang,
                    target_lang: targetLang,
                }, { timeout: 300000 });
                response.data.extracted_text = transcript;
                setResult(response.data);
                setActiveTab('translated');
            } catch (err) {
                console.error(err);
                setError(err.response?.data?.detail || 'Translation failed.');
            } finally {
                setLoading(false);
            }
            return;
        }

        if (inputMode === 'file' && !file) {
            setError('Please upload a document first.');
            return;
        }
        if (isFileReviewStep && !ocrDraftText.trim()) {
            setError('Please review the extracted OCR text before translating.');
            return;
        }
        if ((inputMode === 'text' || inputMode === 'ranjana' || inputMode === 'tamyig') && !inputText.trim()) {
            setError('Please enter or paste some text to translate.');
            return;
        }
        const normalizedRanjanaText = inputMode === 'ranjana'
            ? (ranjanaPreviewReady ? ranjanaPreviewText : convertFromRanjanaLegacy(inputText))
            : "";
        const normalizedTamyigText = inputMode === 'tamyig'
            ? (tamyigPreviewReady ? tamyigPreviewText : convertFromTamyigUnicode(inputText))
            : "";

        if (inputMode === 'ranjana' && !normalizedRanjanaText.trim()) {
            setError('Please paste Ranjana text to convert and translate.');
            return;
        }
        if (inputMode === 'tamyig' && !normalizedTamyigText.trim()) {
            setError('Please paste Tamyig text to convert and translate.');
            return;
        }
        if (inputMode === 'ranjana' && !ranjanaPreviewReady) {
            setRanjanaPreviewText(normalizedRanjanaText);
            setRanjanaPreviewReady(true);
        }
        if (inputMode === 'tamyig' && !tamyigPreviewReady) {
            setTamyigPreviewText(normalizedTamyigText);
            setTamyigPreviewReady(true);
        }

        setLoading(true);
        setError(null);
        if (!isFileReviewStep) {
            setResult(null);
        }
        setUseRanjanaFont(false);
        setUseTamyigFont(false);

        if (window.innerWidth <= 768) {
            setTimeout(() => {
                resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }

        try {
            let response;
            let nextResult;
            let nextTab = 'translated';
            if (inputMode === 'file') {
                if (isFileReviewStep) {
                    response = await axios.post(`${API_BASE_URL}/translate`, {
                        text: ocrDraftText,
                        source_lang: sourceLang,
                        target_lang: targetLang,
                        repair_ocr: true,
                    }, { timeout: 300000 });
                    nextResult = {
                        ...result,
                        ...response.data,
                        extracted_text: ocrDraftText,
                        workflow_stage: 'translated',
                        timing: {
                            ...(result?.timing || {}),
                            ...(response.data.timing || {}),
                            ocr_processing_seconds: result?.timing?.ocr_processing_seconds || 0,
                            total_processing_seconds: Number(
                                ((result?.timing?.ocr_processing_seconds || 0) + (response.data.timing?.total_processing_seconds || 0)).toFixed(2)
                            ),
                        },
                    };
                } else {
                    const formData = new FormData();
                    formData.append('file', file);
                    response = await axios.post(`${API_BASE_URL}/ocrextraction`, formData, {
                        headers: { 'Content-Type': 'multipart/form-data' },
                        timeout: 300000,
                    });
                    nextResult = response.data;
                    setOcrDraftText(response.data.extracted_text || '');
                    nextTab = 'extracted';
                }
            } else {
                const textForTranslation = inputMode === 'ranjana'
                    ? normalizedRanjanaText
                    : inputMode === 'tamyig'
                        ? normalizedTamyigText
                        : inputText;
                const sourceForTranslation = sourceLang === 'Tamang/Newari'
                    ? inputMode === 'ranjana'
                        ? 'Newari'
                        : inputMode === 'tamyig'
                            ? 'Tamang'
                            : sourceLang
                    : sourceLang;

                response = await axios.post(`${API_BASE_URL}/translate`, {
                    text: textForTranslation,
                    source_lang: sourceForTranslation,
                    target_lang: targetLang,
                }, { timeout: 300000 });
                response.data.extracted_text = textForTranslation;
                if (inputMode === 'ranjana') {
                    response.data.original_ranjana_text = inputText;
                }
                if (inputMode === 'tamyig') {
                    response.data.original_tamyig_text = inputText;
                }
                nextResult = response.data;
            }
            setResult(nextResult);
            setActiveTab(nextTab);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || 'An error occurred during processing.');
        } finally {
            setLoading(false);
        }
    };


    const handleToggleRanjana = () => {
        setActiveTab('translated');
        setUseRanjanaFont((current) => {
            const next = !current;
            if (next) setUseTamyigFont(false);
            return next;
        });
    };

    const handleToggleTamyig = () => {
        setActiveTab('translated');
        setUseTamyigFont((current) => {
            const next = !current;
            if (next) setUseRanjanaFont(false);
            return next;
        });
    };

    const toBase64 = (arrayBuffer) => {
        const bytes = new Uint8Array(arrayBuffer);
        let binary = "";
        bytes.forEach((byte) => {
            binary += String.fromCharCode(byte);
        });
        return window.btoa(binary);
    };

    const handleDownloadPDF = async () => {
        if (!result) return;

        try {
            const pdf = new jsPDF({
                orientation: "portrait",
                unit: "mm",
                format: "a4",
            });

            const shouldUseRanjanaPdf = activeTab === "translated" && useRanjanaFont;
            const shouldUseTamyigPdf = activeTab === "translated" && useTamyigFont;
            if (shouldUseRanjanaPdf) {
                const fontResponse = await fetch("/nithya-ranjana.otf");
                if (!fontResponse.ok) throw new Error("Ranjana font could not be loaded.");
                const ranjanaBase64 = toBase64(await fontResponse.arrayBuffer());
                pdf.addFileToVFS("nithya-ranjana.otf", ranjanaBase64);
                pdf.addFont("nithya-ranjana.otf", "Ranjana", "normal");
                pdf.setFont("Ranjana");
            } else if (shouldUseTamyigPdf) {
                const fontResponse = await fetch("/monlam-uni-ouchan5.ttf");
                if (!fontResponse.ok) throw new Error("Tamyig font could not be loaded.");
                const tamyigBase64 = toBase64(await fontResponse.arrayBuffer());
                pdf.addFileToVFS("monlam-uni-ouchan5.ttf", tamyigBase64);
                pdf.addFont("monlam-uni-ouchan5.ttf", "Tamyig", "normal");
                pdf.setFont("Tamyig");
            } else {
                pdf.addFileToVFS("NotoSansDevanagari.ttf", notoDevanagari);
                pdf.addFont("NotoSansDevanagari.ttf", "NotoSans", "normal");
                pdf.setFont("NotoSans");
            }

            pdf.setFontSize(12);
            pdf.setTextColor(0, 0, 0);

            const pageWidth = pdf.internal.pageSize.getWidth();
            const pageHeight = pdf.internal.pageSize.getHeight();
            const margin = 15;
            const maxLineWidth = pageWidth - margin * 2;

            const text = activeTab === "translated"
                ? (
                    useRanjanaFont
                        ? convertToRanjanaLegacy(result.translated_text)
                        : useTamyigFont
                            ? convertToTamyigUnicode(result.translated_text)
                            : result.translated_text
                )
                : isOcrReview
                    ? ocrDraftText
                    : result.extracted_text;

            const lines = pdf.splitTextToSize(text, maxLineWidth);

            let cursorY = 20;

            lines.forEach((line) => {
                if (cursorY > pageHeight - margin) {
                    pdf.addPage();
                    cursorY = margin;
                }
                pdf.text(line, margin, cursorY);
                cursorY += 7;
            });

            pdf.save("translation_result.pdf");

        } catch (err) {
            console.error("PDF generation failed:", err);
            setError("Failed to generate PDF.");
        }
    };



    return (
        <div className="dashboard-container">
            <header className="dashboard-header glass-panel">
                <div className="header-content container">
                    <div className="logo-section">
                        <TamangTranslationIcon className="logo-icon" size={32} />
                        <h1 className="logo-text">Nep<span className="text-accent">Text</span></h1>
                    </div>
                    <div className="user-profile">
                        <div className="avatar">A</div>
                        <span>Admin User</span>
                    </div>
                </div>
            </header>

            <main className="main-content container">
                <section className="hero-section animate-fade-in">
                    <h2 className="hero-title">Intelligent Devanagari OCR & Translation</h2>
                    <p className="hero-description">
                        This system allows you to upload photos and multi-page PDFs containing Devanagari text.
                        Our AI expertly extracts the content and translates it across Himalayan languages,
                        streamlining your document processing workflow.
                    </p>
                </section>

                <div className="split-view">
                    {/* Left Panel: Upload & Preview */}
                    <div className="panel left-panel glass-panel animate-fade-in">
                        <div className="panel-header">
                            <h2><RefreshCw size={20} /> Input & Language</h2>
                        </div>

                        <div className="upload-area">
                            <div className="language-selectors">
                                <div className="selector-group">
                                    <label>Source Language</label>
                                    <select
                                        className="lang-select"
                                        value={sourceLang}
                                        onChange={(e) => setSourceLang(e.target.value)}
                                    >
                                        <option value="Tamang/Newari">Auto-Detect</option>
                                        <option value="Tamang">Tamang</option>
                                        <option value="Newari">Newari (Nepal Bhasa)</option>
                                        <option value="Nepali">Nepali</option>
                                        <option value="English">English</option>
                                    </select>
                                </div>
                                <div className="selector-group">
                                    <label>Target Language</label>
                                    <select
                                        className="lang-select"
                                        value={targetLang}
                                        onChange={(e) => setTargetLang(e.target.value)}
                                    >
                                        <option value="Nepali">Nepali</option>
                                        <option value="Tamang">Tamang</option>
                                        <option value="Nepal Bhasa">Newari (Nepal Bhasa)</option>
                                        <option value="English">English</option>
                                    </select>
                                </div>
                            </div>

                            <div className="mode-toggle">
                                <button
                                    className={`mode-btn ${inputMode === 'file' ? 'active' : ''}`}
                                    onClick={() => setInputMode('file')}
                                >
                                    <Upload size={16} /> File Upload
                                </button>
                                <button
                                    className={`mode-btn ${inputMode === 'text' ? 'active' : ''}`}
                                    onClick={() => setInputMode('text')}
                                >
                                    <Type size={16} /> Text Input
                                </button>
                                <button
                                    className={`mode-btn ${inputMode === 'ranjana' ? 'active' : ''}`}
                                    onClick={() => setInputMode('ranjana')}
                                >
                                    <Type size={16} /> Ranjana Text
                                </button>
                                <button
                                    className={`mode-btn ${inputMode === 'tamyig' ? 'active' : ''}`}
                                    onClick={() => setInputMode('tamyig')}
                                >
                                    <Type size={16} /> Tamyig Text
                                </button>
                                <button
                                    className={`mode-btn ${inputMode === 'audio' ? 'active' : ''}`}
                                    onClick={() => setInputMode('audio')}
                                >
                                    <Mic size={16} /> Audio
                                </button>
                            </div>

                            {inputMode === 'file' ? (
                                <label htmlFor="file-upload" className={`drop-zone ${file ? 'has-file' : ''}`}>
                                    <input
                                        id="file-upload"
                                        type="file"
                                        onChange={handleFileChange}
                                        accept=".jpg,.jpeg,.png,.pdf"
                                        className="hidden-input"
                                    />
                                    <input
                                        id="camera-scan"
                                        type="file"
                                        onChange={handleFileChange}
                                        accept="image/*"
                                        capture="environment"
                                        className="hidden-input"
                                    />
                                    <div className="drop-content">
                                        {file ? (
                                            <div className="file-info">
                                                <FileText size={48} className="text-primary" />
                                                <p className="filename">{file.name}</p>
                                                <p className="filesize">{(file.size / 1024).toFixed(2)} KB</p>
                                                <span className="btn-change">Change File</span>
                                            </div>
                                        ) : (
                                            <>
                                                <div className="upload-actions">
                                                    <div className="upload-main">
                                                        <Upload size={48} className="upload-icon" />
                                                        <p className="upload-text">Drag & drop or browse</p>
                                                        <p className="upload-hint">Supports JPG, PNG, PDF</p>
                                                    </div>

                                                    <div className="camera-scan-option">
                                                        <div className="separator text-secondary">
                                                            <span>OR</span>
                                                        </div>
                                                        <button
                                                            className="btn btn-secondary scan-btn"
                                                            onClick={(e) => {
                                                                e.preventDefault();
                                                                e.stopPropagation();
                                                                // Detection logic: Use native on mobile, custom modal on desktop
                                                                const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
                                                                if (isMobile) {
                                                                    document.getElementById('camera-scan').click();
                                                                } else {
                                                                    startDesktopCamera();
                                                                }
                                                            }}
                                                        >
                                                            <Camera size={20} /> Use Camera
                                                        </button>
                                                    </div>
                                                </div>
                                                <div className="sample-hint">
                                                    No file? Try a sample:
                                                    <span className="sample-link" onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleUseSample('Tamang_Nep.pdf'); }}> Tamang</span>
                                                    {' or '}
                                                    <span className="sample-link" onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleUseSample('Newari_Nep.pdf'); }}>Newari</span>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </label>
                            ) : inputMode === 'audio' ? (
                                <div className="audio-input-container">
                                    {/* ── Engine Selection Dropdown ── */}
                                    <div className="engine-select-row" style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                                        <span className="text-secondary" style={{ fontSize: '0.85rem' }}>Transcription Method:</span>
                                        <select
                                            className="lang-select"
                                            value={selectedEngine}
                                            onChange={(e) => setSelectedEngine(e.target.value)}
                                            style={{
                                                padding: '0.4rem 0.8rem',
                                                fontSize: '0.85rem',
                                                borderRadius: '0.375rem',
                                            }}
                                            disabled={isRecording}
                                        >
                                            <option value="auto">Auto-Fallback Pipeline</option>
                                            {transcriptionModels.map((model) => (
                                                <option key={model.id} value={getModelOptionValue(model)}>
                                                    {getCleanProviderName(model.modelName || model.value)}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                    {/* ── Controls Row ── */}
                                    <div className={`audio-recorder-bar ${isRecording ? 'recording' : ''}`}>
                                        <div className="recorder-left">
                                            <button
                                                className={`record-btn ${isRecording ? 'stop' : 'start'}`}
                                                onClick={isRecording ? stopRecording : startRecording}
                                                disabled={!isRecording && liveStatus === 'connecting'}
                                                title={isRecording ? 'Stop recording' : 'Start recording'}
                                            >
                                                {isRecording ? <Square size={22} /> : <Mic size={22} />}
                                            </button>
                                            <div className="recorder-info">
                                                {isRecording ? (
                                                    <>
                                                        <span className="rec-dot" />
                                                        <span className="rec-time">{formatTime(recordingTime)}</span>
                                                        <span className="rec-label">Recording</span>
                                                    </>
                                                ) : liveStatus === 'done' ? (
                                                    <span className="rec-done">✓ Done — edit transcript below</span>
                                                ) : liveStatus === 'connecting' ? (
                                                    <span className="rec-label">Connecting...</span>
                                                ) : (
                                                    <span className="rec-label">Click mic to start live transcription</span>
                                                )}
                                            </div>
                                        </div>
                                        <div className="recorder-right">
                                            {(transcript || isRecording) && (
                                                <button className="btn btn-ghost btn-sm" onClick={clearAudio} title="Clear & start over">
                                                    <Trash2 size={15} /> Clear
                                                </button>
                                            )}
                                        </div>
                                    </div>

                                    {/* ── WS status pill ── */}
                                    {wsStatusMsg && (
                                        <div className={`ws-status-pill ${liveStatus}`}>
                                            {liveStatus === 'live' && <span className="ws-dot" />}
                                            {wsStatusMsg}
                                        </div>
                                    )}

                                    {/* ── Live transcript / editable area ── */}
                                    {isRecording ? (
                                        // DURING recording: read-only live view
                                        <div className="live-transcript-view" ref={liveViewRef}>
                                            {transcript ? (
                                                <span className="live-transcript-text">{transcript}<span className="live-cursor">▋</span></span>
                                            ) : (
                                                <span className="live-placeholder">Speak now… text will appear here</span>
                                            )}
                                        </div>
                                    ) : transcript ? (
                                        // AFTER recording: editable textarea
                                        <>
                                            <div className="transcript-toolbar">
                                                <span className="transcript-label">Transcript — edit before translating</span>
                                                <div className="transcript-actions">
                                                    <button
                                                        className="btn btn-ghost btn-xs"
                                                        title="Copy transcript"
                                                        onClick={handleCopyTranscript}
                                                    >
                                                        {copied ? (
                                                            <>
                                                                <Check size={14} style={{ color: 'var(--success)' }} /> Copied!
                                                            </>
                                                        ) : (
                                                            <>
                                                                <FileText size={14} /> Copy
                                                            </>
                                                        )}
                                                    </button>
                                                    <button
                                                        className="btn btn-ghost btn-xs"
                                                        title="Download as .txt"
                                                        onClick={() => {
                                                            const blob = new Blob([transcript], { type: 'text/plain' });
                                                            const url = URL.createObjectURL(blob);
                                                            const a = document.createElement('a');
                                                            a.href = url; a.download = 'transcript.txt'; a.click();
                                                            URL.revokeObjectURL(url);
                                                        }}
                                                    >
                                                        <Download size={14} /> .txt
                                                    </button>
                                                </div>
                                            </div>
                                            <textarea
                                                className="transcript-edit-field"
                                                value={transcript}
                                                onChange={(e) => { setTranscript(e.target.value); transcriptRef.current = e.target.value; }}
                                                placeholder="Your transcript will appear here after recording..."
                                            />
                                        </>
                                    ) : (
                                        // EMPTY state
                                        <div className="audio-empty-hint">
                                            <Mic size={36} className="audio-empty-icon" />
                                            <p>Press the mic button to start.<br />Your words will appear here live as you speak.</p>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="text-input-container">
                                    <textarea
                                        className={`text-input-field ${inputMode === 'ranjana' ? 'ranjana-input-field' : ''} ${inputMode === 'tamyig' ? 'tamyig-input-field' : ''}`}
                                        placeholder={
                                            inputMode === 'ranjana'
                                                ? "Paste Ranjana text here..."
                                                : inputMode === 'tamyig'
                                                    ? "Paste Tamyig text here..."
                                                    : "Paste your Tamang or Newari text here..."
                                        }
                                        value={inputText}
                                        onChange={(e) => {
                                            if (inputMode === 'ranjana') {
                                                handleRanjanaInputChange(e.target.value);
                                            } else if (inputMode === 'tamyig') {
                                                handleTamyigInputChange(e.target.value);
                                            } else {
                                                setInputText(e.target.value);
                                            }
                                        }}
                                    ></textarea>
                                    {inputMode === 'ranjana' && (
                                        <div className="ranjana-preview-panel">
                                            <div className="ranjana-preview-header">
                                                <span>Devanagari Preview</span>
                                                <div className="ranjana-preview-actions">
                                                    {DEVANAGARI_PATTERN.test(inputText) && (
                                                        <button
                                                            className="btn btn-secondary btn-xs"
                                                            onClick={encodeDevanagariInputAsRanjana}
                                                            disabled={!inputText.trim()}
                                                            title="Convert copied Devanagari into Ranjana font encoding"
                                                        >
                                                            <Type size={14} /> Encode as Ranjana
                                                        </button>
                                                    )}
                                                    <button
                                                        className="btn btn-primary btn-xs"
                                                        onClick={generateRanjanaPreview}
                                                        disabled={!inputText.trim()}
                                                        title="Show equivalent Devanagari"
                                                    >
                                                        <Type size={14} /> See equivalent Devanagari
                                                    </button>
                                                    <button
                                                        className="btn btn-ghost btn-xs"
                                                        onClick={() => handleRanjanaInputChange('')}
                                                        disabled={!inputText}
                                                        title="Clear Ranjana input"
                                                    >
                                                        <Trash2 size={14} /> Clear
                                                    </button>
                                                    <button
                                                        className="btn btn-ghost btn-xs"
                                                        onClick={copyRanjanaEncodedText}
                                                        disabled={!inputText.trim()}
                                                        title="Copy legacy Ranjana font-code text"
                                                    >
                                                        <FileText size={14} /> Copy Ranjana
                                                    </button>
                                                </div>
                                            </div>
                                            {ranjanaActionMsg && (
                                                <div className="ranjana-action-note">{ranjanaActionMsg}</div>
                                            )}
                                            <div className="ranjana-preview-box">
                                                {ranjanaPreviewReady
                                                    ? (ranjanaPreviewText || "No convertible text found.")
                                                    : "Click the button above to show equivalent Devanagari."}
                                            </div>
                                        </div>
                                    )}
                                    {inputMode === 'tamyig' && (
                                        <div className="ranjana-preview-panel">
                                            <div className="ranjana-preview-header">
                                                <span>Devanagari Preview</span>
                                                <div className="ranjana-preview-actions">
                                                    {DEVANAGARI_PATTERN.test(inputText) && (
                                                        <button
                                                            className="btn btn-secondary btn-xs"
                                                            onClick={encodeDevanagariInputAsTamyig}
                                                            disabled={!inputText.trim()}
                                                            title="Convert copied Devanagari into Tamyig"
                                                        >
                                                            <Type size={14} /> Encode as Tamyig
                                                        </button>
                                                    )}
                                                    <button
                                                        className="btn btn-primary btn-xs"
                                                        onClick={generateTamyigPreview}
                                                        disabled={!inputText.trim()}
                                                        title="Show equivalent Devanagari"
                                                    >
                                                        <Type size={14} /> See equivalent Devanagari
                                                    </button>
                                                    <button
                                                        className="btn btn-ghost btn-xs"
                                                        onClick={() => handleTamyigInputChange('')}
                                                        disabled={!inputText}
                                                        title="Clear Tamyig input"
                                                    >
                                                        <Trash2 size={14} /> Clear
                                                    </button>
                                                    <button
                                                        className="btn btn-ghost btn-xs"
                                                        onClick={copyTamyigEncodedText}
                                                        disabled={!inputText.trim()}
                                                        title="Copy Tamyig text"
                                                    >
                                                        <FileText size={14} /> Copy Tamyig
                                                    </button>
                                                </div>
                                            </div>
                                            {tamyigActionMsg && (
                                                <div className="ranjana-action-note">{tamyigActionMsg}</div>
                                            )}
                                            <div className="ranjana-preview-box">
                                                {tamyigPreviewReady
                                                    ? (tamyigPreviewText || "No convertible text found.")
                                                    : "Click the button above to show equivalent Devanagari."}
                                            </div>
                                        </div>
                                    )}
                                    <div className="input-footer">
                                        <span>
                                            {inputMode === 'ranjana'
                                                ? `Ranjana characters: ${inputText.length}`
                                                : inputMode === 'tamyig'
                                                    ? `Tamyig characters: ${inputText.length}`
                                                    : `Character count: ${inputText.length}`}
                                        </span>
                                        <span className="sample-hint" style={{ border: 'none', width: 'auto', padding: 0 }}>
                                            Need a sample?
                                            <span
                                                className="sample-link"
                                                onClick={() => {
                                                    if (inputMode === 'ranjana') {
                                                        handleRanjanaInputChange(convertToRanjanaLegacy('नेपाल भाषा'));
                                                    } else if (inputMode === 'tamyig') {
                                                        handleTamyigInputChange(convertToTamyigUnicode('छ्याल्हाबा, खन्ता बा तबा मुला?'));
                                                    } else {
                                                        setInputText('छ्याल्हाबा, खन्ता बा तबा मुला?');
                                                    }
                                                }}
                                            >
                                                {inputMode === 'ranjana' ? ' Ranjana' : inputMode === 'tamyig' ? ' Tamyig' : ' Tamang'}
                                            </span>
                                        </span>
                                    </div>
                                </div>
                            )}

                            {previewUrl && (
                                <div className="image-preview">
                                    <img src={previewUrl} alt="Preview" />
                                </div>
                            )}



                            <button
                                className="btn btn-primary w-full mt-4"
                                onClick={handleTranslate}
                                disabled={loading || (inputMode === 'audio' && isRecording)}
                            >
                                {loading ? (
                                    <>
                                        <RefreshCw size={18} className="spin" /> {inputMode === 'file' && !isOcrReview ? 'Extracting OCR...' : 'Translating...'}
                                    </>
                                ) : (
                                    <>
                                        {inputMode === 'audio' ? 'Translate Transcript' : inputMode === 'file' ? (isOcrReview ? 'Translate Reviewed Text' : 'Extract OCR Text') : inputMode === 'ranjana' ? 'Translate Ranjana Text' : inputMode === 'tamyig' ? 'Translate Tamyig Text' : 'Translate Text'} <ArrowRight size={18} />
                                    </>
                                )}
                            </button>

                            {error && (
                                <div className="error-message">
                                    <AlertCircle size={18} />
                                    <span>{error}</span>
                                </div>
                            )}

                            <div className="info-note">
                                <div className="info-note-icon">
                                    <Info size={20} />
                                </div>
                                <div className="info-note-content">
                                    <h4>Model Performance Note</h4>
                                    <p>
                                        As this is a demo environment, we are currently utilizing cost-efficient AI models.
                                        While highly effective, occasional nuances may occur. For enterprise-grade production,
                                        we support integration with premium, high-performance LLMs to ensure the highest
                                        level of accuracy and reliability.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right Panel: Result */}
                    <div className="panel right-panel glass-panel animate-fade-in" style={{ animationDelay: '0.1s' }} ref={resultsRef}>
                        <div className="panel-header">
                            <h2><FileText size={20} /> Results</h2>
                            {result && (
                                <div className="header-actions">
                                    <button
                                        className={`btn ${isReading ? 'btn-secondary' : 'btn-primary'} btn-sm`}
                                        onClick={handleReadAloud}
                                        title={isReading ? "Stop Reading" : "Read Aloud"}
                                        style={{ marginRight: '8px' }}
                                    >
                                        {isReading ? <VolumeX size={16} /> : <Volume2 size={16} />}
                                        {isReading ? " Stop" : " Read Aloud"}
                                    </button>
                                    <button
                                        className="btn btn-primary btn-sm"
                                        onClick={handleDownloadPDF}
                                        title="Download PDF"
                                    >
                                        <Download size={16} /> Download PDF
                                    </button>
                                    {!isOcrReview && (
                                        <>
                                            <button
                                                className={`btn ${useRanjanaFont ? 'btn-secondary' : 'btn-primary'} btn-sm`}
                                                onClick={handleToggleRanjana}
                                                title={useRanjanaFont ? "Show standard font" : "Convert translated text to Ranjana"}
                                            >
                                                <Type size={16} /> {useRanjanaFont ? "Standard Font" : "Convert to Ranjana"}
                                            </button>
                                            <button
                                                className={`btn ${useTamyigFont ? 'btn-secondary' : 'btn-primary'} btn-sm`}
                                                onClick={handleToggleTamyig}
                                                title={useTamyigFont ? "Show standard font" : "Convert translated text to Tamyig"}
                                            >
                                                <Type size={16} /> {useTamyigFont ? "Standard Font" : "Convert to Tamyig"}
                                            </button>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>

                        <div className="result-area">
                            {loading && !result ? (
                                <LoadingState />
                            ) : result ? (
                                <div className="result-content">
                                    {result.ocr_review_required && (
                                        <div className="ocr-quality-warning">
                                            <AlertCircle size={18} />
                                            <span>
                                                {result.ocr_quality?.message
                                                    || 'OCR quality is low. Please review the extracted text before translating.'}
                                            </span>
                                        </div>
                                    )}
                                    <div className="tabs">
                                        <button
                                            className={`tab ${activeTab === 'translated' ? 'active' : ''} ${isOcrReview ? 'disabled' : ''}`}
                                            onClick={() => {
                                                if (!isOcrReview) setActiveTab('translated');
                                            }}
                                            disabled={isOcrReview}
                                        >
                                            <Check size={16} /> Translated Content
                                        </button>
                                        <button
                                            className={`tab ${activeTab === 'extracted' ? 'active' : ''}`}
                                            onClick={() => setActiveTab('extracted')}
                                        >
                                            <FileText size={16} /> {inputMode === 'audio' ? 'Transcript' : inputMode === 'ranjana' || inputMode === 'tamyig' ? 'Normalized Source' : 'Original Text'}
                                        </button>
                                    </div>

                                    <div className={`text-box highlight scrollable ${activeTab === 'translated' && useRanjanaFont ? 'ranjana-text' : ''} ${activeTab === 'translated' && useTamyigFont ? 'tamyig-text' : ''}`}>
                                        {activeTab === 'translated' ? (
                                            <TypewriterText text={translatedDisplayText} />
                                        ) : isOcrReview ? (
                                            <textarea
                                                className="ocr-review-field"
                                                value={ocrDraftText}
                                                onChange={(e) => setOcrDraftText(e.target.value)}
                                                placeholder="Review and correct the extracted OCR text before translating..."
                                            />
                                        ) : (
                                            result.extracted_text
                                        )}
                                    </div>

                                    {/* Results are now clean and focused only on text */}

                                    <div className="result-footer">
                                        <p className={`status-badge ${result.ocr_review_required ? 'warning' : 'success'}`}>
                                            {result.ocr_review_required
                                                ? <AlertCircle size={14} />
                                                : <Check size={14} />}
                                            {result.ocr_review_required
                                                ? 'OCR Needs Careful Review'
                                                : isOcrReview
                                                    ? 'OCR Ready for Review'
                                                    : 'Processing Complete'}
                                        </p>

                                        {result.timing && (
                                            <div className="timing-info animate-fade-in">
                                                <RefreshCw size={12} className="timing-icon" />
                                                {result.timing.ocr_processing_seconds > 0 && (
                                                    <span className="timing-item">OCR: <strong>{result.timing.ocr_processing_seconds}s</strong></span>
                                                )}
                                                {result.timing.transcription_seconds > 0 && (
                                                    <span className="timing-item">Audio: <strong>{result.timing.transcription_seconds}s</strong></span>
                                                )}
                                                {!isOcrReview && (
                                                    <span className="timing-item">AI: <strong>{result.timing.llm_api_response_seconds}s</strong></span>
                                                )}
                                                <span className="timing-item total">Total: <strong>{result.timing.total_processing_seconds}s</strong></span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ) : (
                                <div className="empty-state">
                                    <TamangTranslationIcon size={64} className="empty-icon" />
                                    <p>
                                        {inputMode === 'audio'
                                            ? 'Record audio, edit the transcript, then click "Translate Transcript" to see results here.'
                                            : inputMode === 'file'
                                                ? 'Upload a document to see the translation results here.'
                                                : inputMode === 'ranjana'
                                                    ? 'Paste Ranjana text, review the Devanagari preview, then translate.'
                                                    : inputMode === 'tamyig'
                                                        ? 'Paste Tamyig text, review the Devanagari preview, then translate.'
                                                        : 'Enter text and click translate to see the results here.'}
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>

            {/* Desktop Camera Modal Overlay */}
            {cameraMode && (
                <div className="camera-modal-overlay">
                    <div className="camera-modal glass-panel">
                        <div className="camera-modal-header">
                            <h3><Camera size={20} /> Document Capture</h3>
                            <button className="btn-close" onClick={stopDesktopCamera}>&times;</button>
                        </div>
                        <div className="camera-viewfinder">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                className="webcam-feed"
                            />
                            <div className="scanning-guide"></div>
                        </div>
                        <div className="camera-modal-footer">
                            <button className="btn btn-secondary" onClick={stopDesktopCamera}>Cancel</button>
                            <button className="btn btn-primary capture-btn" onClick={captureFromDesktop}>
                                <div className="shutter-icon"></div> Capture Photo
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Dashboard;
