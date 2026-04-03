/**
 * test_suite.cpp  –  Unit & integration tests (no MPI required).
 *
 * Compile:
 *   g++ -std=c++17 -O2 -DNO_MPI -I../include -o run_tests test_suite.cpp -lstdc++fs
 * Run:
 *   ./run_tests
 */
#include <iostream>
#include <sstream>
#include <cstring>
#include <random>
#include <functional>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "../include/common.hpp"
#include "../include/external_sort.hpp"
#include "../include/verifier.hpp"

namespace fs = std::filesystem;

// ── Framework ─────────────────────────────────────────────────────────────────
struct TR { bool ok; std::string name, detail; };
static std::vector<TR> g_results;

static void run_test(const std::string& name, std::function<void()> fn) {
    std::cout << "  [ RUN ] " << name << " ... " << std::flush;
    bool ok = true; std::string detail;
    try { fn(); }
    catch (const std::exception& e){ ok=false; detail=e.what(); }
    catch (...) { ok=false; detail="Unknown exception"; }
    g_results.push_back({ok,name,detail});
    std::cout << (ok ? "PASS" : "FAIL: "+detail) << "\n";
}
static void CHK(bool c, const std::string& m){ if(!c) throw std::runtime_error("Check: "+m); }
template<typename A,typename B>
static void EQ(A a,B b,const std::string& m){
    if(a!=b){std::ostringstream s;s<<m<<" (got "<<a<<" != "<<b<<")";
        throw std::runtime_error(s.str());}}

// ── Helpers ───────────────────────────────────────────────────────────────────
static Config make_cfg(size_t mb=16){
    Config c; c.mem_limit_mb=mb; c.samples_per_node=100;
    c.merge_batch=512; c.tmp_dir="/tmp/dsort_test2";
    c.verbose=false; c.stable_sort=true; return c;
}
static Record make_rec(uint64_t kv, uint8_t pb=0xAB){
    Record r; memset(&r,0,sizeof(r));
    for(int i=7;i>=0;--i){r.key[i]=char(kv&0xFF);kv>>=8;}
    memset(r.payload,pb,PAYLOAD_SIZE); return r;
}
static std::vector<Record> rand_recs(size_t n, uint64_t seed=42){
    std::mt19937_64 rng(seed);
    std::vector<Record> v(n);
    for(auto& r:v) r=make_rec(rng(),uint8_t(rng()&0xFF));
    return v;
}
static bool sorted_ok(const std::vector<Record>& v){
    for(size_t i=1;i<v.size();++i) if(v[i]<v[i-1]) return false;
    return true;
}
static void wfile(const std::string& p, const std::vector<Record>& v){
    fs::create_directories(fs::path(p).parent_path());
    std::ofstream f(p,std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()),v.size()*RECORD_SIZE);
}
static std::vector<Record> rfile(const std::string& p){
    size_t n=file_record_count(p);
    std::vector<Record> v(n);
    std::ifstream f(p,std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()),n*RECORD_SIZE);
    return v;
}

// ── Tests ─────────────────────────────────────────────────────────────────────
static void test_record(){
    run_test("Record: sizeof==100",[]{ EQ(sizeof(Record),(size_t)100,"sizeof"); });
    run_test("Record: operator<",[]{ auto a=make_rec(1),b=make_rec(2); CHK(a<b,"a<b"); CHK(!(b<a),"!b<a"); });
    run_test("Record: operator==",[]{ auto a=make_rec(7,0xAA),b=make_rec(7,0xFF); CHK(a==b,"key=="); });
    run_test("Record: 0x00<0xFF",[]{ Record lo,hi; memset(lo.key,0,KEY_SIZE); memset(hi.key,0xFF,KEY_SIZE); CHK(lo<hi,"lo<hi"); });
    run_test("Record: operator<= self",[]{ auto a=make_rec(5); CHK(a<=a,"a<=a"); });
}

static void test_mem_sort(){
    auto cfg=make_cfg(16);
    fs::create_directories(cfg.tmp_dir);
    run_test("MemSort: empty",[cfg]{ ExternalSorter s(0,cfg); std::vector<Record> v; s.sort_in_memory(v); CHK(v.empty(),""); });
    run_test("MemSort: 1 rec",[cfg]{ auto v=rand_recs(1); ExternalSorter s(0,cfg); s.sort_in_memory(v); EQ(v.size(),(size_t)1,""); });
    run_test("MemSort: 1k random",[cfg]{ auto v=rand_recs(1000); ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); });
    run_test("MemSort: pre-sorted",[cfg]{ auto v=rand_recs(1000); std::sort(v.begin(),v.end()); ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); });
    run_test("MemSort: reverse",[cfg]{ auto v=rand_recs(1000); std::sort(v.rbegin(),v.rend()); ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); });
    run_test("MemSort: all-dup keys",[cfg]{ std::vector<Record> v(500,make_rec(0x42)); ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); });
    run_test("MemSort: 100k random",[cfg]{ auto v=rand_recs(100000,7); ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); EQ(v.size(),(size_t)100000,""); });
}

static void test_file_sort(){
    auto cfg_small=make_cfg(1);   // 1 MB RAM → many small runs
    auto cfg_med  =make_cfg(4);
    std::string in="/tmp/dsort_test2/fin.bin", out="/tmp/dsort_test2/fout.bin";
    fs::create_directories("/tmp/dsort_test2");

    run_test("FileSort: 10k@1MB (multi-run)",[&]{
        wfile(in,rand_recs(10000));
        ExternalSorter s(0,cfg_small); EQ(s.sort_file(in,out),(size_t)10000,"n");
        auto res=rfile(out); EQ(res.size(),(size_t)10000,"sz"); CHK(sorted_ok(res),"sorted");
    });
    run_test("FileSort: empty",[&]{
        wfile(in,{}); ExternalSorter s(0,cfg_small); EQ(s.sort_file(in,out),(size_t)0,"n");
        EQ(file_record_count(out),(size_t)0,"empty_out");
    });
    run_test("FileSort: 1 record",[&]{
        wfile(in,rand_recs(1)); ExternalSorter s(0,cfg_med); EQ(s.sort_file(in,out),(size_t)1,"n");
        CHK(sorted_ok(rfile(out)),"");
    });
    run_test("FileSort: 50k, 100 distinct keys",[&]{
        std::vector<Record> v(50000); std::mt19937_64 r(99);
        for(auto& rec:v) rec=make_rec(r()%100);
        wfile(in,v); ExternalSorter s(0,cfg_med); s.sort_file(in,out);
        auto res=rfile(out); EQ(res.size(),v.size(),"sz"); CHK(sorted_ok(res),"sorted");
    });
    run_test("FileSort: count preserved (7777)",[&]{
        wfile(in,rand_recs(7777,1234)); ExternalSorter s(0,cfg_small);
        EQ(s.sort_file(in,out),(size_t)7777,"7777");
    });
    // Verify safe_records_per_mb doesn't return 0 at low limits
    run_test("FileSort: safe_records_per_mb(16) > 0",[]{ CHK(safe_records_per_mb(16)>0,""); });

    fs::remove(in); fs::remove(out);
}

static void test_verifier(){
    std::string p="/tmp/dsort_test2/v.bin";
    fs::create_directories("/tmp/dsort_test2");
    run_test("Verifier: sorted passes",[&]{
        auto v=rand_recs(5000); std::sort(v.begin(),v.end()); wfile(p,v);
        CHK(Verifier::verify(p,5000,false).ok,"");
    });
    run_test("Verifier: wrong count fails",[&]{
        auto v=rand_recs(100); std::sort(v.begin(),v.end()); wfile(p,v);
        CHK(!Verifier::verify(p,999,false).ok,"");
    });
    run_test("Verifier: empty OK",[&]{
        wfile(p,{}); CHK(Verifier::verify(p,0,false).ok,"");
    });
    run_test("Verifier: quick_check on sorted",[&]{
        auto v=rand_recs(10000); std::sort(v.begin(),v.end()); wfile(p,v);
        CHK(Verifier::quick_check(p,500),"");
    });
    fs::remove(p);
}

static void test_edge(){
    auto cfg=make_cfg(4);
    fs::create_directories(cfg.tmp_dir);
    run_test("Edge: all 0xFF keys",[cfg]{ std::vector<Record> v; for(int i=0;i<100;++i){Record r;memset(r.key,0xFF,KEY_SIZE);memset(r.payload,i,PAYLOAD_SIZE);v.push_back(r);} ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); });
    run_test("Edge: all 0x00 keys",[cfg]{ std::vector<Record> v; for(int i=0;i<100;++i){Record r;memset(r.key,0x00,KEY_SIZE);memset(r.payload,i,PAYLOAD_SIZE);v.push_back(r);} ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); });
    run_test("Edge: alternating min/max",[cfg]{ std::vector<Record> v; for(int i=0;i<500;++i){Record r;memset(r.key,i%2?0x00:0xFF,KEY_SIZE);memset(r.payload,0,PAYLOAD_SIZE);v.push_back(r);} ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); EQ(v.size(),(size_t)500,""); });
    run_test("Edge: 2 recs reversed",[cfg]{ std::vector<Record> v={make_rec(2),make_rec(1)}; ExternalSorter s(0,cfg); s.sort_in_memory(v); CHK(sorted_ok(v),""); });
    run_test("Edge: file_record_count",[]{
        std::string pp="/tmp/dsort_test2/cnt.bin";
        wfile(pp,rand_recs(1234)); EQ(file_record_count(pp),(size_t)1234,""); fs::remove(pp);
    });
}

int main(){
    std::cout << "\n=== Distributed Sort v2 – Unit Tests ===\n\n";
    std::cout << "── Record ──\n";    test_record();
    std::cout << "\n── In-Memory Sort ──\n"; test_mem_sort();
    std::cout << "\n── File Sort ──\n";      test_file_sort();
    std::cout << "\n── Verifier ──\n";       test_verifier();
    std::cout << "\n── Edge Cases ──\n";     test_edge();

    int ok=0,fail=0;
    for(auto& r:g_results) r.ok?++ok:++fail;
    std::cout << "\n══════════════════════════════════\n";
    std::cout << "  " << ok << " passed";
    if(fail) std::cout << ", " << fail << " FAILED";
    std::cout << " / " << g_results.size() << " total\n";
    std::cout << "══════════════════════════════════\n";
    if(fail){ std::cout << "\nFailed:\n"; for(auto& r:g_results) if(!r.ok) std::cout<<"  ✗ "<<r.name<<"\n    "<<r.detail<<"\n"; }
    fs::remove_all("/tmp/dsort_test2");
    return fail?1:0;
}
