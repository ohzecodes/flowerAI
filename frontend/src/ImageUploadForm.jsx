import React, { useState, useRef } from "react";
import { toast } from "react-toastify";

const ImageUploadForm = ({ onSubmit }) => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const inputRef = useRef(null);

  const handleImageChange = (e) => {
    e.preventDefault();
    const file = e.target.files[0];
    if (file) {
      setImage(file);

      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <form
    style={{height: "fit-content"}}
      className="image-upload-form glass-container"
    >
      <div className="form-group">
        <div
          onClick={() => inputRef.current.click()}
          style={{
            backgroundColor: "rgba(241,241,242, 0.5)",
            padding: 5,
            cursor: "pointer"
          }}
        >
          <input
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            ref={inputRef}
            onChange={handleImageChange}
            onClick={(e) => e.stopPropagation()} // Prevents the form from being triggered
            required
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}
          >
            <button
              style={{
                background: "#333",
                color: "white",
                border: "none",
                padding: "10px 20px",
                cursor: "pointer",
                borderRadius: "5px"
              }}
            >
              Choose
            </button>
            <span style={{ marginLeft: "10px", cursor: "pointer" }}>
              {image ? image.name : "No file chosen"}
              <span className="required">*</span>
            </span>
          </div>
        </div>
      </div>
      {imagePreview && (
        <div onClick={()=>{
             if (inputRef.current)  inputRef.current.value = "";
            setImagePreview(null);
            setImage(null);
            
            setImagePreview(null);
            setImage(null)}}
             class="image-container">
          <img
            
            src={imagePreview}
            className="image-preview"
            alt="Preview"
            height={100}
            style={{ margin: "12px", borderRadius: "5px" }}
          />
          <div class="overlay">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="50"
              height="50"
              fill="white"
              class="bi bi-x-octagon"
              viewBox="0 0 16 16"
            >
              <path d="M4.54.146A.5.5 0 0 1 4.893 0h6.214a.5.5 0 0 1 .353.146l4.394 4.394a.5.5 0 0 1 .146.353v6.214a.5.5 0 0 1-.146.353l-4.394 4.394a.5.5 0 0 1-.353.146H4.893a.5.5 0 0 1-.353-.146L.146 11.46A.5.5 0 0 1 0 11.107V4.893a.5.5 0 0 1 .146-.353zM5.1 1 1 5.1v5.8L5.1 15h5.8l4.1-4.1V5.1L10.9 1z" />
              <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708" />
            </svg>
          </div>
        </div>
      )}
      <button  onClick={(e) => {
        e.preventDefault();
        if(!image)
           return toast("please select an image", { type: "error", className: "error-toast" });
   
        onSubmit(e, image);
        
      }} className="btn-upload">
        Upload
      </button>
    </form>
  );
};

export default ImageUploadForm;
