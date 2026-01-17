#!/bin/bash

API="http://localhost:8000/api/v1"

echo "🧪 Week 2 Complete Test Suite"
echo "=============================="

# Test 1: File Write
echo "Test 1: File Write"
curl -s -X POST "$API/tasks" \
  -H "Content-Type: application/json" \
  -d '{"task": "Use file_write tool to create test_suite.txt with TEST PASSED"}' | python3 -m json.tool
sleep 35

# Test 2: File Read
echo ""
echo "Test 2: File Read"
curl -s -X POST "$API/tasks" \
  -H "Content-Type: application/json" \
  -d '{"task": "Use file_read tool to read test_suite.txt"}' | python3 -m json.tool
sleep 35

# Test 3: File List
echo ""
echo "Test 3: File List"
curl -s -X POST "$API/tasks" \
  -H "Content-Type: application/json" \
  -d '{"task": "Use file_list tool to list all files"}' | python3 -m json.tool
sleep 30

# Verify
echo ""
echo "================================"
echo "📊 VERIFICATION"
echo "================================"

echo ""
echo "1. Workspace files:"
docker-compose exec api ls -la /app/workspace/shared/

echo ""
echo "2. File content:"
docker-compose exec api cat /app/workspace/shared/test_suite.txt

echo ""
echo "3. Memory count:"
docker-compose exec postgres psql -U agent -d agent_system -c "SELECT COUNT(*) FROM memories;"

echo ""
echo "4. Tool count:"
docker-compose logs api | grep "tool_registered" | wc -l

echo ""
echo "✅ Test suite complete!"