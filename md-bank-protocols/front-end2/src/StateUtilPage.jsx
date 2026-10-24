import { useAppState } from "./StateContext";

export default function StateUtilPage() {
    const { state, updateState } = useAppState();

    function addExampleState() {
        updateState({
            exemplo: "State atualizado pela página de utilitário",
        });
    }

    return (
        <section className="flex flex-col gap-4 p-6 text-white">
            <h2 className="text-xl font-bold">Utilitário de State</h2>
            <pre className="rounded bg-gray-800 p-4">
                {JSON.stringify(state, null, 2)}
            </pre>
            <button
                className="w-fit rounded bg-blue-600 px-4 py-2 hover:bg-blue-500"
                onClick={addExampleState}
            >
                Atualizar State
            </button>
        </section>
    );
}
