import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, Mail, ArrowRight } from 'lucide-react';
import './Login.css'; // We'll create a small CSS file or use inline styles for specifics if needed, but index.css covers most.

const Login = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);

    const handleLogin = (e) => {
        e.preventDefault();
        setLoading(true);
        // Simulate login delay for realism
        setTimeout(() => {
            setLoading(false);
            navigate('/dashboard');
        }, 1500);
    };

    return (
        <div className="login-container">
            <div className="login-card glass-panel animate-fade-in">
                <div className="login-header">
                    <h1 className="logo-text">TransLate<span className="text-accent">AI</span></h1>
                    <p className="subtitle">Secure Enterprise Login</p>
                </div>
                
                <form onSubmit={handleLogin} className="login-form">
                    <div className="form-group">
                        <label>Email Address</label>
                        <div className="input-wrapper">
                            <Mail size={20} className="input-icon" />
                            <input type="email" placeholder="admin@company.com" defaultValue="admin@company.com" className="input-field pl-10" />
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Password</label>
                        <div className="input-wrapper">
                            <Lock size={20} className="input-icon" />
                            <input type="password" placeholder="••••••••" defaultValue="password" className="input-field pl-10" />
                        </div>
                    </div>

                    <button type="submit" className="btn btn-primary w-full" disabled={loading}>
                        {loading ? 'Authenticating...' : (
                            <>
                                Sign In <ArrowRight size={18} />
                            </>
                        )}
                    </button>
                    
                    <p className="footer-text">Protected by Enterprise SSO</p>
                </form>
            </div>
        </div>
    );
};

export default Login;
