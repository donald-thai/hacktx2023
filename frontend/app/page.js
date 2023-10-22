'use client'

import Image from "next/image";
import { useRouter } from 'next/navigation'
import { motion } from "framer-motion"

export default function Home() {

  const router = useRouter();

  return (
    <div className="bg-black h-screen w-sceen px-32 py-16">
      <div className=" text-white flex flex-col w-full h-full gap-3">
        <header className="font-bold">
          <h2 className="text-2xl mb-3">COOLPILOT</h2>
          <div className="text-[7rem] leading-none flex gap-4">
            <div>
              <h1>//</h1>
            </div>
            <div>
              <motion.h1 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y:0 }} transition={{duration: 1}} className="">CODE WHAT</motion.h1>
              <h1> <motion.span initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y:0 }} transition={{duration: 1, delay: .8}}>YOU</motion.span> <motion.span className="bg-gradient-to-r from-green-300 via-blue-500 to-purple-600 text-transparent bg-clip-text" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y:0 }} transition={{duration: 1, delay: 1.6}}>IMAGINE.</motion.span></h1>
            </div>
          </div>
        </header>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y:0 }} transition={{duration: 1, delay: 2.5}} className="flex flex-1 gap-8 loading-bottom">
          <div className="flex flex-col flex-1 justify-end gap-4">
            <h3 className="text-3xl font-semibold">AI driven coding assistance without the privacy issues.</h3>  
            <div className="py-3 px-6 bg-[#02484A] text-center w-fit font-bold cursor-pointer" onClick={() => router.push('/generator')}>GET STARTED</div>
          </div>

          <div className="w-1/2 max-h-full overflow-hidden relative">
            <Image
              src={"/grid.jpg"}
              alt="turquoise grid"
              fill
              style={{ height: "100%", width: "100%", objectFit: "cover" }}
            />
          </div>
        </motion.div>
      </div>
    </div>
  );
}
