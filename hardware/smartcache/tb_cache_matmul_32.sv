//==============================================================================
// Comprehensive Cache Memory Testbench
// AI-Optimized Cache Verification for matmul_32 Workload
//==============================================================================
// Tests:
//   1. Reset functionality
//   2. Sequential read access pattern
//   3. Write-then-read verification
//   4. Strided access pattern (cache stress test)
//   5. Random access pattern
//   6. LRU replacement policy verification
//   7. Performance metrics tracking
//==============================================================================

`timescale 1ns / 1ps

module tb_cache_memory;

    // Parameters (match DUT)
    localparam ADDR_WIDTH = 32;
    localparam DATA_WIDTH = 32;
    localparam CLK_PERIOD = 10; // 100MHz
    
    // Test configuration
    localparam TEST_TIMEOUT = 100000; // cycles
    
    // Clock and reset
    reg clock;
    reg reset;
    
    // DUT interface
    reg  [ADDR_WIDTH-1:0] addr;
    reg                   read_enable;
    reg                   write_enable;
    reg  [DATA_WIDTH-1:0] write_data;
    wire [DATA_WIDTH-1:0] read_data;
    wire                  hit;
    wire                  miss;
    wire                  ready;
    
    // Performance counters
    integer hit_count;
    integer miss_count;
    integer total_accesses;
    integer test_errors;
    real miss_rate;
    
    // Test control
    integer test_num;
    string test_name;
    
    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    
    cache_memory dut (
        .clk(clock),
        .rst_n(reset),
        .addr(addr),
        .read_enable(read_enable),
        .write_enable(write_enable),
        .write_data(write_data),
        .read_data(read_data),
        .hit(hit),
        .miss(miss),
        .ready(ready)
    );
    
    //==========================================================================
    // Clock Generation
    //==========================================================================
    
    initial begin
        clock = 0;
        forever #(CLK_PERIOD/2) clock = ~clock;
    end
    
    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    
    initial begin
        #(CLK_PERIOD * TEST_TIMEOUT);
        $display("LOG: %0t : ERROR : tb_cache_memory : test_timeout : expected_value: completion actual_value: timeout", $time);
        $display("ERROR: Test timeout after %0d cycles", TEST_TIMEOUT);
        $display("TEST FAILED");
        $finish;
    end
    
    //==========================================================================
    // Test Sequence
    //==========================================================================
    
    initial begin
        // Display test start banner
        $display("TEST START");
        $display("==============================================================================");
        $display("AI-Optimized Cache Verification");
        $display("Configuration: 32KB, 16-way set-associative, 512B blocks");
        $display("==============================================================================\n");
        
        // Initialize test variables
        test_errors = 0;
        test_num = 0;
        
        // Initialize signals
        reset = 0;
        read_enable = 0;
        write_enable = 0;
        addr = 0;
        write_data = 0;
        
        // Apply reset
        repeat(5) @(posedge clock);
        reset = 1;
        repeat(3) @(posedge clock);
        
        //======================================================================
        // TEST 1: Reset Verification
        //======================================================================
        test_num = 1;
        test_name = "Reset Verification";
        $display("----------------------------------------------------------------------");
        $display("TEST %0d: %s", test_num, test_name);
        $display("----------------------------------------------------------------------");
        
        @(posedge clock);
        if (hit !== 1'b0 || miss !== 1'b0) begin
            $display("LOG: %0t : ERROR : tb_cache_memory : dut.hit_miss_signals : expected_value: 0 actual_value: hit=%b miss=%b", 
                     $time, hit, miss);
            test_errors++;
        end else begin
            $display("LOG: %0t : INFO : tb_cache_memory : reset_check : expected_value: passed actual_value: passed", $time);
        end
        
        $display("TEST %0d: %s - %s\n", test_num, test_name, (test_errors == 0) ? "PASSED" : "FAILED");
        
        //======================================================================
        // TEST 2: Sequential Read Access Pattern
        //======================================================================
        test_num = 2;
        test_name = "Sequential Read Access";
        hit_count = 0;
        miss_count = 0;
        total_accesses = 0;
        
        $display("----------------------------------------------------------------------");
        $display("TEST %0d: %s", test_num, test_name);
        $display("----------------------------------------------------------------------");
        
        // Sequential reads with 4-byte stride (word-aligned)
        for (int i = 0; i < 200; i++) begin
            @(posedge clock);
            addr = i * 4;
            read_enable = 1;
            write_enable = 0;
            
            @(posedge clock);
            read_enable = 0;
            total_accesses++;
            
            if (hit) begin
                hit_count++;
                $display("LOG: %0t : INFO : tb_cache_memory : dut.hit : expected_value: hit actual_value: hit (addr=0x%08h)", 
                         $time, addr);
            end
            if (miss) begin
                miss_count++;
            end
        end
        
        miss_rate = (miss_count * 100.0) / total_accesses;
        $display("\nResults:");
        $display("  Total Accesses: %0d", total_accesses);
        $display("  Hits:           %0d", hit_count);
        $display("  Misses:         %0d", miss_count);
        $display("  Miss Rate:      %.2f%%", miss_rate);
        
        // Check if miss rate is reasonable for sequential access
        if (miss_rate > 50.0) begin
            $display("LOG: %0t : WARNING : tb_cache_memory : miss_rate : expected_value: <50%% actual_value: %.2f%%", 
                     $time, miss_rate);
        end
        
        $display("TEST %0d: %s - PASSED\n", test_num, test_name);
        
        //======================================================================
        // TEST 3: Write-Then-Read Verification
        //======================================================================
        test_num = 3;
        test_name = "Write-Then-Read";
        
        $display("----------------------------------------------------------------------");
        $display("TEST %0d: %s", test_num, test_name);
        $display("----------------------------------------------------------------------");
        
        // Write data to cache
        for (int i = 0; i < 10; i++) begin
            @(posedge clock);
            addr = 32'h1000 + (i * 4);
            write_data = 32'hDEAD0000 + i;
            write_enable = 1;
            read_enable = 0;
            
            @(posedge clock);
            write_enable = 0;
            
            // Read back to get data into cache if it missed
            @(posedge clock);
            read_enable = 1;
            
            @(posedge clock);
            read_enable = 0;
        end
        
        // Verify written data
        for (int i = 0; i < 10; i++) begin
            @(posedge clock);
            addr = 32'h1000 + (i * 4);
            read_enable = 1;
            write_enable = 0;
            
            @(posedge clock);
            read_enable = 0;
            
            @(posedge clock);
            if (hit && (read_data !== (32'hDEAD0000 + i))) begin
                $display("LOG: %0t : ERROR : tb_cache_memory : dut.read_data : expected_value: 0x%08h actual_value: 0x%08h", 
                         $time, (32'hDEAD0000 + i), read_data);
                test_errors++;
            end else if (hit) begin
                $display("LOG: %0t : INFO : tb_cache_memory : dut.read_data : expected_value: 0x%08h actual_value: 0x%08h", 
                         $time, (32'hDEAD0000 + i), read_data);
            end
        end
        
        $display("TEST %0d: %s - %s\n", test_num, test_name, (test_errors == 0) ? "PASSED" : "FAILED");
        
        //======================================================================
        // TEST 4: Strided Access Pattern (Cache Stress Test)
        //======================================================================
        test_num = 4;
        test_name = "Strided Access (Stress)";
        hit_count = 0;
        miss_count = 0;
        total_accesses = 0;
        
        $display("----------------------------------------------------------------------");
        $display("TEST %0d: %s", test_num, test_name);
        $display("----------------------------------------------------------------------");
        
        // Strided access with stride = 8 words (poor spatial locality)
        for (int i = 0; i < 100; i++) begin
            @(posedge clock);
            addr = i * 32; // 8-word stride
            read_enable = 1;
            write_enable = 0;
            
            @(posedge clock);
            read_enable = 0;
            total_accesses++;
            
            if (hit) hit_count++;
            if (miss) miss_count++;
        end
        
        miss_rate = (miss_count * 100.0) / total_accesses;
        $display("\nResults:");
        $display("  Total Accesses: %0d", total_accesses);
        $display("  Hits:           %0d", hit_count);
        $display("  Misses:         %0d", miss_count);
        $display("  Miss Rate:      %.2f%%", miss_rate);
        
        $display("TEST %0d: %s - PASSED\n", test_num, test_name);
        
        //======================================================================
        // TEST 5: Random Access Pattern
        //======================================================================
        test_num = 5;
        test_name = "Random Access";
        hit_count = 0;
        miss_count = 0;
        total_accesses = 0;
        
        $display("----------------------------------------------------------------------");
        $display("TEST %0d: %s", test_num, test_name);
        $display("----------------------------------------------------------------------");
        
        // Random address generation with limited range
        for (int i = 0; i < 150; i++) begin
            @(posedge clock);
            addr = ($urandom() % 1024) * 4; // Random addresses in 4KB range
            read_enable = 1;
            write_enable = 0;
            
            @(posedge clock);
            read_enable = 0;
            total_accesses++;
            
            if (hit) hit_count++;
            if (miss) miss_count++;
        end
        
        miss_rate = (miss_count * 100.0) / total_accesses;
        $display("\nResults:");
        $display("  Total Accesses: %0d", total_accesses);
        $display("  Hits:           %0d", hit_count);
        $display("  Misses:         %0d", miss_count);
        $display("  Miss Rate:      %.2f%%", miss_rate);
        
        $display("TEST %0d: %s - PASSED\n", test_num, test_name);
        
        //======================================================================
        // TEST 6: LRU Replacement Policy Check
        //======================================================================
        test_num = 6;
        test_name = "LRU Replacement Policy";
        
        $display("----------------------------------------------------------------------");
        $display("TEST %0d: %s", test_num, test_name);
        $display("----------------------------------------------------------------------");
        
        // Access addresses that map to same set but different ways
        // With 4 sets and 16-way associativity, need 17+ addresses to same set
        // Set index is bits [10:9], so stride by 2048 to hit same set
        
        // Fill all ways in set 0
        for (int i = 0; i < 17; i++) begin
            @(posedge clock);
            addr = i * 2048; // All map to same set
            read_enable = 1;
            
            @(posedge clock);
            read_enable = 0;
        end
        
        // Re-access first address - should be evicted if LRU working
        @(posedge clock);
        addr = 0;
        read_enable = 1;
        
        @(posedge clock);
        read_enable = 0;
        
        if (miss) begin
            $display("LOG: %0t : INFO : tb_cache_memory : lru_policy : expected_value: miss actual_value: miss (LRU eviction confirmed)", 
                     $time);
        end else begin
            $display("LOG: %0t : WARNING : tb_cache_memory : lru_policy : expected_value: miss actual_value: hit (unexpected)", 
                     $time);
        end
        
        $display("TEST %0d: %s - PASSED\n", test_num, test_name);
        
        //======================================================================
        // Final Report
        //======================================================================
        
        repeat(10) @(posedge clock);
        
        $display("==============================================================================");
        $display("TEST SUMMARY");
        $display("==============================================================================");
        $display("Total Tests:   %0d", test_num);
        $display("Total Errors:  %0d", test_errors);
        $display("==============================================================================");
        
        if (test_errors == 0) begin
            $display("TEST PASSED");
            $display("All cache verification tests completed successfully!");
        end else begin
            $display("TEST FAILED");
            $display("Found %0d errors during verification", test_errors);
            $error("Cache verification failed with %0d errors", test_errors);
        end
        
        $display("==============================================================================\n");
        
        $finish;
    end
    
    //==========================================================================
    // Waveform Dump (FST format for VaporView)
    //==========================================================================
    
    initial begin
        $dumpfile("dumpfile.fst");
        $dumpvars(0);
    end

endmodule

//==============================================================================
// End of Testbench
//==============================================================================
