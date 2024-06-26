/**
 * @file lldxhardware.cpp
 * @brief LLDXHardware implementation
 *
 * $LicenseInfo:firstyear=2001&license=viewerlgpl$
 * Second Life Viewer Source Code
 * Copyright (C) 2010, Linden Research, Inc.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License only.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Linden Research, Inc., 945 Battery Street, San Francisco, CA  94111  USA
 * $/LicenseInfo$
 */

#ifdef LL_WINDOWS

// Culled from some Microsoft sample code

#include "linden_common.h"

#define INITGUID
#include <dxdiag.h>
#undef INITGUID

#include <dxgi.h>
#include <wbemidl.h>
#include <comdef.h>

#include <boost/tokenizer.hpp>

#include "lldxhardware.h"

#include "llerror.h"

#include "llstring.h"
#include "llstl.h"
#include "lltimer.h"

void (*gWriteDebug)(const char* msg) = nullptr;
LLDXHardware gDXHardware;

//-----------------------------------------------------------------------------
// Defines, and constants
//-----------------------------------------------------------------------------
#define SAFE_DELETE(p)       { if(p) { delete (p);     (p)=NULL; } }
#define SAFE_DELETE_ARRAY(p) { if(p) { delete[] (p);   (p)=NULL; } }
#define SAFE_RELEASE(p)      { if(p) { (p)->Release(); (p)=NULL; } }

typedef BOOL ( WINAPI* PfnCoSetProxyBlanket )( IUnknown* pProxy, DWORD dwAuthnSvc, DWORD dwAuthzSvc,
                                               OLECHAR* pServerPrincName, DWORD dwAuthnLevel, DWORD dwImpLevel,
                                               RPC_AUTH_IDENTITY_HANDLE pAuthInfo, DWORD dwCapabilities );

//Getting the version of graphics controller driver via WMI
std::string LLDXHardware::getDriverVersionWMI(EGPUVendor vendor)
{
    std::string mDriverVersion;
    HRESULT hres;
    CoInitializeEx(0, COINIT_APARTMENTTHREADED);
    IWbemLocator *pLoc = NULL;

    hres = CoCreateInstance(
        CLSID_WbemLocator,
        0,
        CLSCTX_INPROC_SERVER,
        IID_IWbemLocator, (LPVOID *)&pLoc);

    if (FAILED(hres))
    {
        LL_DEBUGS("AppInit") << "Failed to initialize COM library. Error code = 0x" << hres << LL_ENDL;
        return std::string();                  // Program has failed.
    }

    IWbemServices *pSvc = NULL;

    // Connect to the root\cimv2 namespace with
    // the current user and obtain pointer pSvc
    // to make IWbemServices calls.
    hres = pLoc->ConnectServer(
        _bstr_t(L"ROOT\\CIMV2"), // Object path of WMI namespace
        NULL,                    // User name. NULL = current user
        NULL,                    // User password. NULL = current
        0,                       // Locale. NULL indicates current
        NULL,                    // Security flags.
        0,                       // Authority (e.g. Kerberos)
        0,                       // Context object
        &pSvc                    // pointer to IWbemServices proxy
        );

    if (FAILED(hres))
    {
        LL_WARNS("AppInit") << "Could not connect. Error code = 0x" << hres << LL_ENDL;
        pLoc->Release();
        CoUninitialize();
        return std::string();                // Program has failed.
    }

    LL_DEBUGS("AppInit") << "Connected to ROOT\\CIMV2 WMI namespace" << LL_ENDL;

    // Set security levels on the proxy -------------------------
    hres = CoSetProxyBlanket(
        pSvc,                        // Indicates the proxy to set
        RPC_C_AUTHN_WINNT,           // RPC_C_AUTHN_xxx
        RPC_C_AUTHZ_NONE,            // RPC_C_AUTHZ_xxx
        NULL,                        // Server principal name
        RPC_C_AUTHN_LEVEL_CALL,      // RPC_C_AUTHN_LEVEL_xxx
        RPC_C_IMP_LEVEL_IMPERSONATE, // RPC_C_IMP_LEVEL_xxx
        NULL,                        // client identity
        EOAC_NONE                    // proxy capabilities
        );

    if (FAILED(hres))
    {
        LL_WARNS("AppInit") << "Could not set proxy blanket. Error code = 0x" << hres << LL_ENDL;
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return std::string();               // Program has failed.
    }
    IEnumWbemClassObject* pEnumerator = NULL;

    // Get the data from the query
    ULONG uReturn = 0;
    hres = pSvc->ExecQuery(
        bstr_t("WQL"),
        bstr_t("SELECT * FROM Win32_VideoController"), //Consider using Availability to filter out disabled controllers
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
        NULL,
        &pEnumerator);

    if (FAILED(hres))
    {
        LL_WARNS("AppInit") << "Query for operating system name failed." << " Error code = 0x" << hres << LL_ENDL;
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return std::string();               // Program has failed.
    }

    while (pEnumerator)
    {
        IWbemClassObject *pclsObj = NULL;
        HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1,
            &pclsObj, &uReturn);

        if (0 == uReturn)
        {
            break;               // If quantity less then 1.
        }

        if (vendor != GPU_ANY)
        {
            VARIANT vtCaptionProp;
            // Might be preferable to check "AdapterCompatibility" here instead of caption.
            hr = pclsObj->Get(L"Caption", 0, &vtCaptionProp, 0, 0);

            if (FAILED(hr))
            {
                LL_WARNS("AppInit") << "Query for Caption property failed." << " Error code = 0x" << hr << LL_ENDL;
                pSvc->Release();
                pLoc->Release();
                CoUninitialize();
                return std::string();               // Program has failed.
            }

            // use characters in the returned driver version
            BSTR caption(vtCaptionProp.bstrVal);

            //convert BSTR to std::string
            std::wstring ws(caption, SysStringLen(caption));
            std::string caption_str(ws.begin(), ws.end());
            LLStringUtil::toLower(caption_str);

            bool found = false;
            switch (vendor)
            {
            case GPU_INTEL:
                found = caption_str.find("intel") != std::string::npos;
                break;
            case GPU_NVIDIA:
                found = caption_str.find("nvidia") != std::string::npos;
                break;
            case GPU_AMD:
                found = caption_str.find("amd") != std::string::npos
                        || caption_str.find("ati ") != std::string::npos
                        || caption_str.find("radeon") != std::string::npos;
                break;
            default:
                break;
            }

            if (found)
            {
                VariantClear(&vtCaptionProp);
            }
            else
            {
                VariantClear(&vtCaptionProp);
                pclsObj->Release();
                continue;
            }
        }

        VARIANT vtVersionProp;

        // Get the value of the DriverVersion property
        hr = pclsObj->Get(L"DriverVersion", 0, &vtVersionProp, 0, 0);

        if (FAILED(hr))
        {
            LL_WARNS("AppInit") << "Query for DriverVersion property failed." << " Error code = 0x" << hr << LL_ENDL;
            pSvc->Release();
            pLoc->Release();
            CoUninitialize();
            return std::string();               // Program has failed.
        }

        // use characters in the returned driver version
        BSTR driverVersion(vtVersionProp.bstrVal);

        //convert BSTR to std::string
        std::wstring ws(driverVersion, SysStringLen(driverVersion));
        std::string str(ws.begin(), ws.end());
        LL_INFOS("AppInit") << " DriverVersion : " << str << LL_ENDL;

        if (mDriverVersion.empty())
        {
            mDriverVersion = str;
        }
        else if (mDriverVersion != str)
        {
            if (vendor == GPU_ANY)
            {
                // Expected from systems with gpus from different vendors
                LL_INFOS("DriverVersion") << "Multiple video drivers detected. Version of second driver: " << str << LL_ENDL;
            }
            else
            {
                // Not Expected!
                LL_WARNS("DriverVersion") << "Multiple video drivers detected from same vendor. Version of second driver : " << str << LL_ENDL;
            }
        }

        VariantClear(&vtVersionProp);
        pclsObj->Release();
    }

    // Cleanup
    // ========
    if (pSvc)
    {
        pSvc->Release();
    }
    if (pLoc)
    {
        pLoc->Release();
    }
    if (pEnumerator)
    {
        pEnumerator->Release();
    }

    // supposed to always call CoUninitialize even if init returned false
    CoUninitialize();

    return mDriverVersion;
}

void get_wstring(IDxDiagContainer* containerp, const WCHAR* wszPropName, wchar_t* wszPropValue, int outputSize)
{
    HRESULT hr;
    VARIANT var;

    VariantInit( &var );
    hr = containerp->GetProp(wszPropName, &var );
    if( SUCCEEDED(hr) )
    {
        // Switch off the type.  There's 4 different types:
        switch( var.vt )
        {
            case VT_UI4:
                swprintf(wszPropValue, outputSize, L"%lu", var.ulVal);  /* Flawfinder: ignore */
                break;
            case VT_I4:
                swprintf(wszPropValue, outputSize, L"%li", var.lVal);   /* Flawfinder: ignore */
                break;
            case VT_BOOL:
                wcscpy_s( wszPropValue, outputSize, (var.boolVal) ? L"true" : L"false");    /* Flawfinder: ignore */
                break;
            case VT_BSTR:
                wcsncpy_s( wszPropValue, outputSize, var.bstrVal, outputSize-1 );   /* Flawfinder: ignore */
                wszPropValue[outputSize-1] = 0;
                break;
        }
    }
    // Clear the variant (this is needed to free BSTR memory)
    VariantClear( &var );
}

std::string get_string(IDxDiagContainer *containerp, const WCHAR *wszPropName)
{
    wchar_t wszPropValue[256];
    get_wstring(containerp, wszPropName, wszPropValue, _countof(wszPropValue));

    return ll_convert_wide_to_string(wszPropValue);
}


LLVersion::LLVersion()
{
    mValid = FALSE;
    S32 i;
    for (i = 0; i < 4; i++)
    {
        mFields[i] = 0;
    }
}

BOOL LLVersion::set(const std::string &version_string)
{
    S32 i;
    for (i = 0; i < 4; i++)
    {
        mFields[i] = 0;
    }
    // Split the version string.
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep(".", "", boost::keep_empty_tokens);
    tokenizer tokens(version_string, sep);


    S32 count = 0;
    for (auto iter = tokens.begin(); iter != tokens.end() && count < 4; ++iter)
    {
        mFields[count] = atoi(iter->c_str());
        count++;
    }
    if (count < 4)
    {
        //LL_WARNS() << "Potentially bogus version string!" << version_string << LL_ENDL;
        for (i = 0; i < 4; i++)
        {
            mFields[i] = 0;
        }
        mValid = FALSE;
    }
    else
    {
        mValid = TRUE;
    }
    return mValid;
}

S32 LLVersion::getField(const S32 field_num)
{
    if (!mValid)
    {
        return -1;
    }
    else
    {
        return mFields[field_num];
    }
}

std::string LLDXDriverFile::dump()
{
    if (gWriteDebug)
    {
        gWriteDebug("Filename:");
        gWriteDebug(mName.c_str());
        gWriteDebug("\n");
        gWriteDebug("Ver:");
        gWriteDebug(mVersionString.c_str());
        gWriteDebug("\n");
        gWriteDebug("Date:");
        gWriteDebug(mDateString.c_str());
        gWriteDebug("\n");
    }
    LL_INFOS() << mFilepath << LL_ENDL;
    LL_INFOS() << mName << LL_ENDL;
    LL_INFOS() << mVersionString << LL_ENDL;
    LL_INFOS() << mDateString << LL_ENDL;

    return "";
}

LLDXDevice::~LLDXDevice()
{
    for_each(mDriverFiles.begin(), mDriverFiles.end(), DeletePairedPointer());
    mDriverFiles.clear();
}

std::string LLDXDevice::dump()
{
    if (gWriteDebug)
    {
        gWriteDebug("StartDevice\n");
        gWriteDebug("DeviceName:");
        gWriteDebug(mName.c_str());
        gWriteDebug("\n");
        gWriteDebug("PCIString:");
        gWriteDebug(mPCIString.c_str());
        gWriteDebug("\n");
    }
    LL_INFOS() << LL_ENDL;
    LL_INFOS() << "DeviceName:" << mName << LL_ENDL;
    LL_INFOS() << "PCIString:" << mPCIString << LL_ENDL;
    LL_INFOS() << "Drivers" << LL_ENDL;
    LL_INFOS() << "-------" << LL_ENDL;
    for (driver_file_map_t::iterator iter = mDriverFiles.begin(),
             end = mDriverFiles.end();
         iter != end; iter++)
    {
        LLDXDriverFile *filep = iter->second;
        filep->dump();
    }
    if (gWriteDebug)
    {
        gWriteDebug("EndDevice\n");
    }

    return "";
}

LLDXDriverFile *LLDXDevice::findDriver(const std::string &driver)
{
    for (driver_file_map_t::iterator iter = mDriverFiles.begin(),
             end = mDriverFiles.end();
         iter != end; iter++)
    {
        LLDXDriverFile *filep = iter->second;
        if (!utf8str_compare_insensitive(filep->mName,driver))
        {
            return filep;
        }
    }

    return nullptr;
}

LLDXHardware::LLDXHardware()
{
    mVRAM = 0;
    gWriteDebug = nullptr;
}

void LLDXHardware::cleanup()
{
}

BOOL LLDXHardware::updateVRAM()
{
    U64 vram = getMBVideoMemoryViaDXGI();

    if (vram > 0)
    {
        mVRAM = vram / (1024 * 1024);
        return TRUE;
    }
    return FALSE;
}

U64 LLDXHardware::getMBVideoMemoryViaDXGI()
{
    HRESULT hr;
    HRESULT hrCoInitialize = S_OK;
    U64 vram = 0;

    hrCoInitialize = CoInitialize(0);
    if (SUCCEEDED(hrCoInitialize))
    {
        IDXGIFactory1* pDXGIFactory = nullptr;

        hr = CreateDXGIFactory1(IID_PPV_ARGS(&pDXGIFactory));
        if (SUCCEEDED(hr))
        {
            llassert(pDXGIFactory != 0);
            IDXGIAdapter1* dxgiAdapter = nullptr;
            IDXGIAdapter1* tmpDxgiAdapter = nullptr;
            UINT adapterIndex = 0;
            while (pDXGIFactory->EnumAdapters1(adapterIndex, &tmpDxgiAdapter) != DXGI_ERROR_NOT_FOUND)
            {
                DXGI_ADAPTER_DESC1 desc;
                hr = tmpDxgiAdapter->GetDesc1(&desc);
                if (SUCCEEDED(hr) && !dxgiAdapter && desc.Flags == 0 && tmpDxgiAdapter)
                {
                    tmpDxgiAdapter->QueryInterface(IID_PPV_ARGS(&dxgiAdapter));
                }
                SAFE_RELEASE(tmpDxgiAdapter);
                ++adapterIndex;
            }
            if (dxgiAdapter != nullptr)
            {
                DXGI_ADAPTER_DESC1 desc;
                dxgiAdapter->GetDesc1(&desc);
                vram = desc.DedicatedVideoMemory;

                SAFE_RELEASE(dxgiAdapter);
            }

            SAFE_RELEASE(pDXGIFactory);
        }
        CoUninitialize();
    }
    return vram;
}

LLSD LLDXHardware::getDisplayInfo()
{
    LLTimer hw_timer;
    HRESULT       hr;
    LLSD ret;
    hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    IDxDiagProvider *dx_diag_providerp = NULL;
    IDxDiagContainer *dx_diag_rootp = NULL;
    IDxDiagContainer *devices_containerp = NULL;
    IDxDiagContainer *device_containerp = NULL;
    IDxDiagContainer *file_containerp = NULL;
    IDxDiagContainer *driver_containerp = NULL;
    DWORD dw_device_count;

    if (FAILED(hr))
    {
        LL_WARNS() << "COM initialization failure!" << LL_ENDL;
        gWriteDebug("COM initialization failure!\n");
        goto LCleanup;
    }

    // CoCreate a IDxDiagProvider*
    LL_INFOS() << "CoCreateInstance IID_IDxDiagProvider" << LL_ENDL;
    hr = CoCreateInstance(CLSID_DxDiagProvider,
                          nullptr,
                          CLSCTX_INPROC_SERVER,
                          IID_IDxDiagProvider,
                          (LPVOID*) &dx_diag_providerp);

    if (FAILED(hr))
    {
        LL_WARNS() << "No DXDiag provider found!  DirectX 9 not installed!" << LL_ENDL;
        gWriteDebug("No DXDiag provider found!  DirectX 9 not installed!\n");
        goto LCleanup;
    }
    if (SUCCEEDED(hr)) // if FAILED(hr) then dx9 is not installed
    {
        // Fill out a DXDIAG_INIT_PARAMS struct and pass it to IDxDiagContainer::Initialize
        // Passing in TRUE for bAllowWHQLChecks, allows dxdiag to check if drivers are
        // digital signed as logo'd by WHQL which may connect via internet to update
        // WHQL certificates.
        DXDIAG_INIT_PARAMS dx_diag_init_params;
        ZeroMemory(&dx_diag_init_params, sizeof(DXDIAG_INIT_PARAMS));

        dx_diag_init_params.dwSize                  = sizeof(DXDIAG_INIT_PARAMS);
        dx_diag_init_params.dwDxDiagHeaderVersion   = DXDIAG_DX9_SDK_VERSION;
        dx_diag_init_params.bAllowWHQLChecks        = TRUE;
        dx_diag_init_params.pReserved               = NULL;

        LL_INFOS() << "dx_diag_providerp->Initialize" << LL_ENDL;
        hr = dx_diag_providerp->Initialize(&dx_diag_init_params);
        if(FAILED(hr))
        {
            goto LCleanup;
        }

        LL_INFOS() << "dx_diag_providerp->GetRootContainer" << LL_ENDL;
        hr = dx_diag_providerp->GetRootContainer( &dx_diag_rootp );
        if(FAILED(hr) || !dx_diag_rootp)
        {
            goto LCleanup;
        }

        HRESULT hr;

        // Get display driver information
        LL_INFOS() << "dx_diag_rootp->GetChildContainer" << LL_ENDL;
        hr = dx_diag_rootp->GetChildContainer(L"DxDiag_DisplayDevices", &devices_containerp);
        if(FAILED(hr) || !devices_containerp)
        {
            // do not release 'dirty' devices_containerp at this stage, only dx_diag_rootp
            devices_containerp = NULL;
            goto LCleanup;
        }

        // make sure there is something inside
        hr = devices_containerp->GetNumberOfChildContainers(&dw_device_count);
        if (FAILED(hr) || dw_device_count == 0)
        {
            goto LCleanup;
        }

        // Get device 0
        LL_INFOS() << "devices_containerp->GetChildContainer" << LL_ENDL;
        hr = devices_containerp->GetChildContainer(L"0", &device_containerp);
        if(FAILED(hr) || !device_containerp)
        {
            goto LCleanup;
        }

        // Get the English VRAM string
        std::string ram_str = get_string(device_containerp, L"szDisplayMemoryEnglish");


        // Dump the string as an int into the structure
        char *stopstring;
        ret["VRAM"] = (LLSD::Integer)strtol(ram_str.c_str(), &stopstring, 10);
        std::string device_name = get_string(device_containerp, L"szDescription");
        ret["DeviceName"] = device_name;
        std::string device_driver=  get_string(device_containerp, L"szDriverVersion");
        ret["DriverVersion"] = device_driver;

        // ATI has a slightly different version string
        if(device_name.length() >= 4 && device_name.substr(0,4) == "ATI ")
        {
            // get the key
            HKEY hKey;
            const DWORD RV_SIZE = 100;
            WCHAR release_version[RV_SIZE];

            // Hard coded registry entry.  Using this since it's simpler for now.
            // And using EnumDisplayDevices to get a registry key also requires
            // a hard coded Query value.
            if(ERROR_SUCCESS == RegOpenKey(HKEY_LOCAL_MACHINE, TEXT("SOFTWARE\\ATI Technologies\\CBT"), &hKey))
            {
                // get the value
                DWORD dwType = REG_SZ;
                DWORD dwSize = sizeof(WCHAR) * RV_SIZE;
                if(ERROR_SUCCESS == RegQueryValueEx(hKey, TEXT("ReleaseVersion"),
                    nullptr, &dwType, (LPBYTE)release_version, &dwSize))
                {
                    // print the value
                    // windows doesn't guarantee to be null terminated
                    release_version[RV_SIZE - 1] = NULL;
                    ret["DriverVersion"] = ll_convert_wide_to_string(release_version);

                }
                RegCloseKey(hKey);
            }
        }
    }

LCleanup:
    if (!ret.isMap() || (ret.size() == 0))
    {
        LL_INFOS() << "Failed to get data, cleaning up" << LL_ENDL;
    }
    SAFE_RELEASE(file_containerp);
    SAFE_RELEASE(driver_containerp);
    SAFE_RELEASE(device_containerp);
    SAFE_RELEASE(devices_containerp);
    SAFE_RELEASE(dx_diag_rootp);
    SAFE_RELEASE(dx_diag_providerp);

    CoUninitialize();
    return ret;
}

void LLDXHardware::setWriteDebugFunc(void (*func)(const char*))
{
    gWriteDebug = func;
}

#endif
