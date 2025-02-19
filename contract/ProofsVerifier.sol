//SPDX-License-Identifier: MIT
pragma solidity >=0.8.19;

import "./plonk_vs.sol";

contract ProofsVerifier {

    UltraVerifier verifier;

    constructor(address _verifier) {
        verifier = UltraVerifier(_verifier);
    }

    function verifyInput(bytes calldata proof, int[] memory inputs) public returns (bool) {
        bytes32[] memory publicInputs = new bytes32[](inputs.length());
        for (uint i = 0; i < inputs.length(); i++) {
            publicInputs[i] = bytes32(inputs[i]);
        }
        require(verifier.verify(proof, publicInputs), "Invalid proof");

        return true;
    }

    function verifyHidden(bytes calldata proof) public returns (bool) {
        bytes32[] memory publicInputs;
        require(verifier.verify(proof, publicInputs), "Invalid proof");

        return true;
    }

    function verifyOutput(bytes calldata proof, int[] memory outputs) public returns (bool) {
        bytes32[] memory publicInputs = new bytes32[](outputs.length());
        for (uint i = 0; i < outputs.length(); i++) {
            publicInputs[i] = bytes32(inputs[i]);
        }
        require(verifier.verify(proof, publicInputs), "Invalid proof");

        return true;
    }

    function verifyAllLayers(bytes[] calldata proofs, int[] memory inputs, int[] memory outputs) public returns (bool) {
        require(verifyInput(proofs[0], inputs), "Invalid proof for input layer");
        for (uint i = 1; i < proofs.length() - 1; i++) {
            require(verifyHidden(proofs[i]), "Invalid proof for hidden layer");
        }
        require(verifyOutput(proofs[proofs.length() - 1], outputs), "Invalid proof for ouput layer");
    }

}
