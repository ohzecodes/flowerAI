import "./App.css";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import React, { useState } from "react";
import HorizontalBarChart from "./BarChart";
import ImageUploadForm from "./ImageUploadForm";

const BackendEndpoint =  process.env.REACT_APP_BACKEND_URL;
console.log(BackendEndpoint);
const FLowerSVG = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    {...props}
    fill="currentColor"
    className="bi bi-flower2"
    viewBox="0 0 16 16"
  >
    <path d="M8 16a4 4 0 0 0 4-4 4 4 0 0 0 0-8 4 4 0 0 0-8 0 4 4 0 1 0 0 8 4 4 0 0 0 4 4m3-12q0 .11-.03.247c-.544.241-1.091.638-1.598 1.084A3 3 0 0 0 8 5c-.494 0-.96.12-1.372.331-.507-.446-1.054-.843-1.597-1.084A1 1 0 0 1 5 4a3 3 0 0 1 6 0m-.812 6.052A3 3 0 0 0 11 8a3 3 0 0 0-.812-2.052c.215-.18.432-.346.647-.487C11.34 5.131 11.732 5 12 5a3 3 0 1 1 0 6c-.268 0-.66-.13-1.165-.461a7 7 0 0 1-.647-.487m-3.56.617a3 3 0 0 0 2.744 0c.507.446 1.054.842 1.598 1.084q.03.137.03.247a3 3 0 1 1-6 0q0-.11.03-.247c.544-.242 1.091-.638 1.598-1.084m-.816-4.721A3 3 0 0 0 5 8c0 .794.308 1.516.812 2.052a7 7 0 0 1-.647.487C4.66 10.869 4.268 11 4 11a3 3 0 0 1 0-6c.268 0 .66.13 1.165.461.215.141.432.306.647.487M8 9a1 1 0 1 1 0-2 1 1 0 0 1 0 2" />
  </svg>
);
const LoadingComponent = () => (
  <div
    style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center"
    }}
  >
    <FLowerSVG width={50} height={50} />
    <div className="ellipsis-animation">
      Loading<span className="ellipsis">...</span>
    </div>
  </div>
);

const TopBar = ({ text }) => {
  const [showAnnouncement, setShowAnnouncement] = useState(true);
  return (
    <>
      {showAnnouncement && (
        <div
          style={{
            width: "100%",
            padding: 10,
            backgroundColor: "rgb(69, 147, 170)",
            justifyContent: "space-around",
            display: "flex",
            color: "white",
            alignItems: "center"
          }}
        >
          <p style={{ color: "inherit" }}> {text}</p>
          <button
            className="close-button"
            onClick={() => setShowAnnouncement(false)}
          >
            &times;
          </button>
        </div>
      )}
    </>
  );
};
function App() {
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const handleSubmit = async (e, image) => {
    e.preventDefault();
    if (!image) return;

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("file", image);
      const response = await fetch(`${BackendEndpoint}/predict`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) throw new Error("Network response was not ok");
      const data = await response.json();
      setImageData(data);
    } catch (error) {
      toast(error.message, { type: "error", className: "error-toast" });
      setImageData(null);
    } finally {
      setLoading(false);
    }
  };


  return (
    <div
      className="App"
      style={{ height: imageData ? "fit-content" : "100vh" }}
    >
      {imageData && (
        <TopBar
          text={`this app uses ${imageData.arch}! A lightweight model, So the predictions may not be perfect`}
        />
      )}
      <header className="App-header">
        <FLowerSVG width={50} height={50} />
        <h1>Flower AI</h1>
      </header>
      <main
        style={{
          justifyContent: "left",
          display: "flex",
          paddingBottom: 20,
          height: "80vh"
        }}
      >
        <ImageUploadForm onSubmit={handleSubmit} />
        <div
          className="glass-container"
          style={{ width: "60vw", padding: 20, margin: 20 }}
        >
          {loading ? (
            <LoadingComponent />
          ) : (
            imageData && <HorizontalBarChart data={imageData} />
          )}
        </div>
      </main>
      <footer>
        <p>© Ohzecodes {new Date().getFullYear()} | Flower AI</p>
      </footer>
      <ToastContainer />
    </div>
  );
}

export default App;
