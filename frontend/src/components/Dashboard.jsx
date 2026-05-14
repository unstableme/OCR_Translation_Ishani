import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Check, AlertCircle, RefreshCw, ArrowRight, Download, Info, Type, Camera, Mic, Square, Trash2 } from 'lucide-react';
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
    const [inputMode, setInputMode] = useState('file'); // 'file' | 'text'
    const [inputText, setInputText] = useState('');
    const [cameraMode, setCameraMode] = useState(false);
    
    // Audio State
    const [isRecording, setIsRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState(null);
    const [recordingTime, setRecordingTime] = useState(0);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const timerRef = useRef(null);

    // Live Transcription State
    const wsRef = useRef(null);
    const [transcript, setTranscript] = useState('');       // live + final transcript
    const [liveStatus, setLiveStatus] = useState('idle');   // idle | connecting | live | processing | done | error
    const [wsStatusMsg, setWsStatusMsg] = useState('');
    const transcriptRef = useRef('');                       // mirror of transcript for WS callbacks
    const liveViewRef = useRef(null);

    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const resultsRef = useRef(null);

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

    // Audio Recording Logic — Live Transcription via WebSocket
    const startRecording = async () => {
        setError(null);
        setTranscript('');
        transcriptRef.current = '';
        setLiveStatus('connecting');
        setWsStatusMsg('Connecting...');

        if (!window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            setError('Microphone access requires a secure connection (HTTPS).');
            setLiveStatus('error');
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setError('Your browser does not support audio recording.');
            setLiveStatus('error');
            return;
        }

        try {
            // 1. Open WebSocket FIRST
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.hostname}:8000/ws/transcribe?lang=${encodeURIComponent(sourceLang)}`;
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                setLiveStatus('live');
                setWsStatusMsg('Listening...');
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'segment' && msg.text) {
                        // Server sends the FULL transcript so far on each chunk
                        transcriptRef.current = msg.text;
                        setTranscript(msg.text);
                        setWsStatusMsg('Listening...');
                        // Auto-scroll live view
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
                } catch (_) { /* ignore parse errors */ }
            };

            ws.onerror = () => {
                setError('WebSocket error. Check that the backend is running.');
                setLiveStatus('error');
            };

            ws.onclose = () => {
                if (liveStatus !== 'done') setLiveStatus('idle');
            };

            // 2. Start microphone
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Prefer webm/opus which browsers output natively
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            const mediaRecorder = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            // Send each timed chunk as binary over WS
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 100 && ws.readyState === WebSocket.OPEN) {
                    audioChunksRef.current.push(event.data);
                    event.data.arrayBuffer().then((buf) => {
                        // Send the raw blob bytes
                        ws.send(buf);
                    }).catch(() => {});
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunksRef.current, { type: mimeType });
                setAudioBlob(blob);
                stream.getTracks().forEach((t) => t.stop());
                // Signal end
                if (ws.readyState === WebSocket.OPEN) ws.send('done');
            };

            // Fire ondataavailable every 3 seconds
            mediaRecorder.start(3000);
            setIsRecording(true);
            setRecordingTime(0);

            timerRef.current = setInterval(() => {
                setRecordingTime((prev) => prev + 1);
            }, 1000);

        } catch (err) {
            console.error('Audio recording error:', err);
            setError('Could not access microphone. Please grant microphone permissions.');
            setLiveStatus('error');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            clearInterval(timerRef.current);
            setLiveStatus('done');
        }
    };

    const clearAudio = () => {
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
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
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
            setError(null);

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
        } catch (err) {
            console.error(err);
            setError(`Could not load ${sampleName} document.`);
        } finally {
            setLoading(false);
        }
    };

    const handleTranslate = async () => {
        if (targetLang !== 'Nepali') {
            setError(`${targetLang} translation is currently unavailable. We are working on it!`);
            return;
        }

        // Audio mode: translate the live transcript text
        if (inputMode === 'audio') {
            if (!transcript.trim()) {
                setError('Please record audio first to get a transcript, then click Translate.');
                return;
            }
            setLoading(true);
            setError(null);
            setResult(null);
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
        if (inputMode === 'text' && !inputText.trim()) {
            setError('Please enter or paste some text to translate.');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        if (window.innerWidth <= 768) {
            setTimeout(() => {
                resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }

        try {
            let response;
            if (inputMode === 'file') {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('source_lang', sourceLang);
                formData.append('target_lang', targetLang);
                response = await axios.post(`${API_BASE_URL}/upload`, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: 300000,
                });
            } else {
                response = await axios.post(`${API_BASE_URL}/translate`, {
                    text: inputText,
                    source_lang: sourceLang,
                    target_lang: targetLang,
                }, { timeout: 300000 });
                response.data.extracted_text = inputText;
            }
            setResult(response.data);
            setActiveTab('translated');
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || 'An error occurred during processing.');
        } finally {
            setLoading(false);
        }
    };


    const handleDownloadPDF = () => {
        if (!result) return;

        try {
            const pdf = new jsPDF({
                orientation: "portrait",
                unit: "mm",
                format: "a4",
            });

            // Register font
            pdf.addFileToVFS("NotoSansDevanagari.ttf", notoDevanagari);
            pdf.addFont("NotoSansDevanagari.ttf", "NotoSans", "normal");
            pdf.setFont("NotoSans");

            pdf.setFontSize(12);
            pdf.setTextColor(0, 0, 0);

            const pageWidth = pdf.internal.pageSize.getWidth();
            const pageHeight = pdf.internal.pageSize.getHeight();
            const margin = 15;
            const maxLineWidth = pageWidth - margin * 2;

            const text = activeTab === "translated"
                ? result.translated_text
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
                                                        onClick={() => navigator.clipboard.writeText(transcript)}
                                                    >
                                                        <FileText size={14} /> Copy
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
                                            <p>Press the mic button to start.<br/>Your words will appear here live as you speak.</p>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="text-input-container">
                                    <textarea
                                        className="text-input-field"
                                        placeholder="Paste your Tamang or Newari text here..."
                                        value={inputText}
                                        onChange={(e) => setInputText(e.target.value)}
                                    ></textarea>
                                    <div className="input-footer">
                                        <span>Character count: {inputText.length}</span>
                                        <span className="sample-hint" style={{ border: 'none', width: 'auto', padding: 0 }}>
                                            Need a sample? 
                                            <span className="sample-link" onClick={() => setInputText('छ्याल्हाबा, खन्ता बा तबा मुला?')}> Tamang</span>
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
                                        <RefreshCw size={18} className="spin" /> Translating...
                                    </>
                                ) : (
                                    <>
                                        {inputMode === 'audio' ? 'Translate Transcript' : inputMode === 'file' ? 'Translate Document' : 'Translate Text'} <ArrowRight size={18} />
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
                                        className="btn btn-primary btn-sm"
                                        onClick={handleDownloadPDF}
                                        title="Download PDF"
                                    >
                                        <Download size={16} /> Download PDF
                                    </button>
                                </div>
                            )}
                        </div>

                        <div className="result-area">
                            {loading && !result ? (
                                <LoadingState />
                            ) : result ? (
                                <div className="result-content">
                                    <div className="tabs">
                                        <button
                                            className={`tab ${activeTab === 'translated' ? 'active' : ''}`}
                                            onClick={() => setActiveTab('translated')}
                                        >
                                            <Check size={16} /> Translated Content
                                        </button>
                                        <button
                                            className={`tab ${activeTab === 'extracted' ? 'active' : ''}`}
                                            onClick={() => setActiveTab('extracted')}
                                        >
                                            <FileText size={16} /> {inputMode === 'audio' ? 'Transcript' : 'Original Text'}
                                        </button>
                                    </div>

                                    <div className="text-box highlight scrollable">
                                        {activeTab === 'translated' ? (
                                            <TypewriterText text={result.translated_text} />
                                        ) : (
                                            result.extracted_text
                                        )}
                                    </div>
                                    
                                    {/* Results are now clean and focused only on text */}

                                    <div className="result-footer">
                                        <p className="status-badge success">
                                            <Check size={14} /> Processing Complete
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
                                                <span className="timing-item">AI: <strong>{result.timing.llm_api_response_seconds}s</strong></span>
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
