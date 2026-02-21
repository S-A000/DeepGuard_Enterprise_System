import React from 'react';

const Navbar = () => {
    return (
        <nav style={{ 
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '10px 30px', backgroundColor: '#1e293b', borderBottom: '1px solid #334155'
        }}>
            <h2 style={{ color: '#38bdf8', margin: 0 }}>DeepGuard <span style={{ fontSize: '12px', color: '#94a3b8' }}>v1.0</span></h2>
            <div style={{ display: 'flex', gap: '20px', color: '#cbd5e1', fontSize: '14px' }}>
                <span>System Status: <b style={{ color: '#22c55e' }}>Online</b></span>
                <span>Mode: <b>Enterprise AI</b></span>
            </div>
        </nav>
    );
};

export default Navbar;