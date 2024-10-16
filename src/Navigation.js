import React from 'react';
import './styles/Navigation.css';
import Dropdown from 'react-bootstrap/Dropdown';
import DropdownButton from 'react-bootstrap/DropdownButton';

const actions = [
    { id: 1, text: "My profile", icon: "user" },
    { id: 2, text: "Messages", icon: "email" },
    { id: 3, text: "Contacts", icon: "group" },
    { id: 4, text: "Log out", icon: "runner" }
];

function Navigation() {
    return (
        <div className="App">
            <nav className="navbar">
                <DropdownButton id="dropdown-basic-button" title="Dropdown button">
                    <Dropdown.Item href="#/action-1">Action</Dropdown.Item>
                    <Dropdown.Item href="#/action-2">Another action</Dropdown.Item>
                    <Dropdown.Item href="#/action-3">Something else</Dropdown.Item>
                </DropdownButton>
            </nav>
        </div>
    );
}

export default Navigation;
