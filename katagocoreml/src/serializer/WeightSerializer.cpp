// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025

#include "WeightSerializer.hpp"
#include "MILBlob/Blob/StorageWriter.hpp"
#include "MILBlob/Util/Span.hpp"

namespace katagocoreml {

size_t WeightSerializer::serialize(std::vector<WeightEntry>& weights,
                                   const std::string& blob_path) {
    MILBlob::Blob::StorageWriter writer(blob_path, true);
    size_t total_bytes = 0;

    for (auto& entry : weights) {
        // Create span from weight data
        MILBlob::Util::Span<const float> span(entry.data.data(), entry.data.size());

        // Write to blob file and record offset
        entry.blob_offset = writer.WriteData(span);

        total_bytes += entry.data.size() * sizeof(float);
    }

    return total_bytes;
}

}  // namespace katagocoreml
