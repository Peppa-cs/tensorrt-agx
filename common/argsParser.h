/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#include <string>
#include <vector>
#ifdef _MSC_VER
#include "..\common\windows\getopt.h"
#else
#include <getopt.h>
#endif
#include <iostream>

namespace samplesCommon
{

//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
struct SampleParams
{
    int batchSize{1};                     //!< Number of inputs in a batch
    int dlaCore{-1};                   //!< Specify the DLA core to run network on.
    int imagenums{1};
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    bool verbose{true};

    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//!
//! \brief The CaffeSampleParams structure groups the additional parameters required by
//!         networks that use caffe
//!
struct CaffeSampleParams : public SampleParams
{
    std::string prototxtFileName; //!< Filename of prototxt design file of a network
    std::string weightsFileName;  //!< Filename of trained weights file of a network
    std::string meanFileName;     //!< Filename of mean file of a network
    std::string modelFileName;     //!< Filename of mean file of a network
    std::string imageFileName;
    std::string referenceFileName;
    std::string lableFileName;
    std::string Network;
    std::string engine;
    std::string calibfile;
};


//!
//! \brief The OnnxSampleParams structure groups the additional parameters required by
//!         networks that use ONNX
//!
struct OnnxSampleParams : public SampleParams
{
    std::string onnxFileName; //!< Filename of ONNX file of a network
    std::string modelFileName;     //!< Filename of mean file of a network
    std::string imageFileName;
    std::string referenceFileName;
    std::string lableFileName;
};

//!
//! \brief The UffSampleParams structure groups the additional parameters required by
//!         networks that use Uff
//!
struct UffSampleParams : public SampleParams
{
    std::string uffFileName; //!< Filename of uff file of a network
};

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInInt8{false};
    bool runInFp16{false};
    bool help{false};
    int useDLACore{-1};
    int batch{1};
    int imagenums{1}; 
    std::string dataDirs;
    std::string prototxtFileName;
    std::string weightsFileName;
    std::string Network;
    std::string engine;
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(Args& args, int argc, char* argv[])
{
    while (1)
    {
        int arg;
        static struct option long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"datadir", required_argument, 0, 'd'},
            {"prototxtFileName", required_argument, 0, 'p'},
            {"weightsFileName", required_argument, 0, 'w'},
            {"int8", no_argument, 0, 'i'},
            {"fp16", no_argument, 0, 'f'},
            {"batch",required_argument,0,'b'},
            {"Network",required_argument,0,'n'},
            {"useDLACore", required_argument, 0, 'u'},
            {"engine", required_argument, 0, 'e'},
            {"calibfile", required_argument, 0, 'c'}
            {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h':
            args.help = true;
            return true;
        case 'p':
            if (optarg)
            {
                args.prototxtFileName.assign(optarg);
            }
            else
            {
                std::cerr << "ERROR: --prototxtFileName requires option argument" << std::endl;
                return false;
            }
            break;
        case 'w':
            if (optarg)
            {
                args.weightsFileName.assign(optarg);
            }
            else
            {
                std::cerr << "ERROR: --weightsFileName requires option argument" << std::endl;
                return false;
            }
            break;
        case 'd':
            if (optarg)
            {
                args.dataDirs.assign(optarg);
            }
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 'i':
            args.runInInt8 = true;
            break;
        case 'f':
            args.runInFp16 = true;
            break;
        case 'u':
            if (optarg)
            {
                args.useDLACore = std::stoi(optarg);
            }
            break;
        case 'b':
            if (optarg)
            {
               args.batch = std::stoi(optarg);
            }
            break;
	case 'n':
            if (optarg)
            {
                args.Network.assign(optarg);
            }
            else
            {
                std::cerr << "ERROR: --network requires option argument" << std::endl;
                return false;
            }
            break;
	case 'c':
            if (optarg)
            {
                args.calibfile.assign(optarg);
            }
            else
            {
                std::cerr << "ERROR: --calibfile requires option argument" << std::endl;
                return false;
            }
            break;
	case 'e':
            if (optarg)
            {
                args.engine.assign(optarg);
            }
            else
            {
                std::cerr << "ERROR: --engine requires option argument" << std::endl;
                return false;
            }
            break;
        default:
            return false;
        }
    }
    return true;
}

} // namespace samplesCommon

#endif // TENSORRT_ARGS_PARSER_H
