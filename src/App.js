import React from 'react';
import './styles/App.css';

function App() {
    return (
        <div className="App">
            <nav className="navbar">
                <div className="dropdown">
                    <button className="dropbtn">Menu 1</button>
                    <div className="dropdown-content">
                        <button>Option 1</button>
                        <button>Option 2</button>
                        <button>Option 3</button>
                        <button>Option 4</button>
                        <button>Option 5</button>
                    </div>
                </div>

                <div className="dropdown">
                    <button className="dropbtn">Menu 2</button>
                    <div className="dropdown-content">
                        <button>Option 1</button>
                        <button>Option 2</button>
                        <button>Option 3</button>
                        <button>Option 4</button>
                        <button>Option 5</button>
                    </div>
                </div>

                <div className="dropdown">
                    <button className="dropbtn">Menu 3</button>
                    <div className="dropdown-content">
                        <button>Option 1</button>
                        <button>Option 2</button>
                        <button>Option 3</button>
                        <button>Option 4</button>
                        <button>Option 5</button>
                    </div>
                </div>
            </nav>
        </div>
    );
}

export default App;
