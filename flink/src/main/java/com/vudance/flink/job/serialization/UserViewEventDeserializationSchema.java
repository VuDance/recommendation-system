package com.vudance.flink.job.serialization;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.vudance.flink.job.model.UserViewEvent;
import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;

import java.io.IOException;

/**
 * JSON deserializer for Kafka user-view-events.
 */
public class UserViewEventDeserializationSchema implements DeserializationSchema<UserViewEvent> {
    private static final long serialVersionUID = 1L;
    private transient ObjectMapper mapper;

    private ObjectMapper getMapper() {
        if (mapper == null) {
            mapper = new ObjectMapper();
        }
        return mapper;
    }

    @Override
    public UserViewEvent deserialize(byte[] bytes) throws IOException {
        return getMapper().readValue(bytes, UserViewEvent.class);
    }

    @Override
    public boolean isEndOfStream(UserViewEvent userViewEvent) {
        return false;
    }

    @Override
    public TypeInformation<UserViewEvent> getProducedType() {
        return TypeInformation.of(UserViewEvent.class);
    }
}
