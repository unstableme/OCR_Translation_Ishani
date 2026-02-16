import React, { useState, useRef } from 'react';
import { Upload, FileText, Check, AlertCircle, RefreshCw, ArrowRight, Download } from 'lucide-react';
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
    "Translating Tamang to Nepali...",
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
                <p className="loading-hint">This usually takes about 30 seconds</p>
            </div>
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

    const resultsRef = useRef(null);

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
            } else {
                setPreviewUrl(null);
            }
        }
    };

    const handleUseSample = async () => {
        setLoading(true);
        setError(null);
        try {
            // Now fetching from the frontend's own public folder
            const response = await fetch('/Tamang_Nep.pdf');
            if (!response.ok) throw new Error('Failed to load sample document');

            const blob = await response.blob();
            const sampleFile = new File([blob], 'Tamang_Nep.pdf', { type: 'application/pdf' });

            setFile(sampleFile);
            setPreviewUrl(null);
            setResult(null);
        } catch (err) {
            console.error(err);
            setError('Could not load sample document.');
        } finally {
            setLoading(false);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please upload a document or use the sample document first.');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        // Scroll to results on mobile
        if (window.innerWidth <= 768) {
            setTimeout(() => {
                resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
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
                        <h1 className="logo-text">TransLate<span className="text-accent">AI</span></h1>
                    </div>
                    <div className="user-profile">
                        <div className="avatar">A</div>
                        <span>Admin User</span>
                    </div>
                </div>
            </header>

            <main className="main-content container">
                <div className="split-view">
                    {/* Left Panel: Upload & Preview */}
                    <div className="panel left-panel glass-panel animate-fade-in">
                        <div className="panel-header">
                            <h2><Upload size={20} /> Document Upload</h2>
                        </div>

                        <div className="upload-area">
                            <label htmlFor="file-upload" className={`drop-zone ${file ? 'has-file' : ''}`}>
                                <input
                                    id="file-upload"
                                    type="file"
                                    onChange={handleFileChange}
                                    accept=".jpg,.jpeg,.png,.pdf"
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
                                            <Upload size={48} className="upload-icon" />
                                            <p className="upload-text">Drag & drop or browse</p>
                                            <p className="upload-hint">Supports JPG, PNG, PDF</p>
                                            <div className="sample-hint" onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleUseSample(); }}>
                                                No file? <span className="sample-link">Use this sample document</span>
                                            </div>
                                        </>
                                    )}
                                </div>
                            </label>

                            {previewUrl && (
                                <div className="image-preview">
                                    <img src={previewUrl} alt="Preview" />
                                </div>
                            )}

                            <button
                                className="btn btn-primary w-full mt-4"
                                onClick={handleUpload}
                                disabled={loading}
                            >
                                {loading ? (
                                    <>
                                        <RefreshCw size={18} className="spin" /> Processing...
                                    </>
                                ) : (
                                    <>
                                        Translate Document <ArrowRight size={18} />
                                    </>
                                )}
                            </button>

                            {error && (
                                <div className="error-message">
                                    <AlertCircle size={18} />
                                    <span>{error}</span>
                                </div>
                            )}
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
                                            <Check size={16} /> Nepali Translation
                                        </button>
                                        <button
                                            className={`tab ${activeTab === 'extracted' ? 'active' : ''}`}
                                            onClick={() => setActiveTab('extracted')}
                                        >
                                            <FileText size={16} /> Original Text
                                        </button>
                                    </div>

                                    <div className="text-box highlight scrollable">
                                        {activeTab === 'translated' ? result.translated_text : result.extracted_text}
                                    </div>

                                    <div className="result-footer">
                                        <p className="status-badge success">
                                            <Check size={14} /> Processing Complete
                                        </p>
                                    </div>
                                </div>
                            ) : (
                                <div className="empty-state">
                                    <TamangTranslationIcon size={64} className="empty-icon" />
                                    <p>Upload a document to see the translation results here.</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Dashboard;
