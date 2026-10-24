import { createContext, useContext, useState } from "react";

const StateContext = createContext(null);

export function StateProvider({ children }) {
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [state, setState] = useState({});
    const [stateVisible, setStateVisible] = useState(false);

    function updateState(nextState) {
        setState((currentState) => ({
            ...currentState,
            ...nextState,
        }));
    }

    return (
        <StateContext.Provider
            value={{
                messages,
                setMessages,
                loading,
                setLoading,
                state,
                updateState,
                stateVisible,
                setStateVisible,
            }}
        >
            {children}
        </StateContext.Provider>
    );
}

export function useAppState() {
    const context = useContext(StateContext);

    if (!context) {
        throw new Error("useAppState deve ser usado dentro de StateProvider");
    }

    return context;
}
