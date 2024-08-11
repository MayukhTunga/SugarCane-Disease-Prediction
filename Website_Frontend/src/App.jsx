import Navbar from "./Navbar.jsx"
import ImageUploader from './ImageUploader';

export default function App() {
  return (
    <>
      <div className="bg-[url('assets/Background.jpg')] bg-cover bg-center h-screen w-full">
      <Navbar/>
      <p className="text-center text-white text-3xl mt-40 font-semibold">Upload Leaf Image</p>
      <ImageUploader/>
      </div>
    </>
  )
}