// SPDX-License-Identifier: GPL-3.0
pragma solidity>=0.6.10 <0.8.20;
pragma experimental ABIEncoderV2;

import "./SafeMath.sol";
import "./Roles.sol";

contract Ownable {
    address public owner;

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev The Ownable constructor sets the original `owner` of the contract to the sender
     * account.
     */
    constructor() public {
        owner = msg.sender;
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        require(msg.sender == owner, "Caller does not have the owner");
        _;
    }

    /**
     * @dev Allows the current owner to transfer control of the contract to a newOwner.
     * @param newOwner The address to transfer ownership to.
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0));
        emit  OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}

contract IssuerRole is Ownable{
    using Roles for Roles.Role;

    event IssuerAdded(address indexed account);
    event IssuerRemoved(address indexed account);

    Roles.Role private _issuers;

    constructor() public {
        _addIssuer(msg.sender);
    }

    modifier onlyIssuer() {
        require(isIssuer(msg.sender), "Caller does not have the Issuer role");
        _;
    }

    function isIssuer(address account) public view returns (bool) {
        return _issuers.has(account);
    }

    function removeIssuer(address account) public onlyOwner {
        _removeIssuer(account);
    }

    // function renounceIssuer() public {
    //     _removeIssuer(msg.sender);
    // }

    function _addIssuer(address account) internal {
        _issuers.add(account);
        emit IssuerAdded(account);
    }

    function _removeIssuer(address account) internal {
        _issuers.remove(account);
        emit IssuerRemoved(account);
    }
}

contract Cred is IssuerRole{
    using SafeMath for uint;

    mapping(address => uint256) private _balances;
    mapping(address => uint256) private _cred;

    mapping(uint256 => int256[128][]) private _globalGradient;
    mapping(uint256 => mapping(uint256 => int256[128][])) private _globalGradientList;

    event UpdateGlobalGradient( address indexed addr, int256[][] value);

    constructor() public {
        _balances[msg.sender] = 100;
        _cred[msg.sender] = 100;
    }

    function register(address account) public onlyOwner {
        _addIssuer(account);
        _balances[account] = 100;
        _cred[account] = 100;
    }

    function shareData(uint round,int256[128][] memory matrix) public onlyIssuer returns (uint, int, int) {
        int val = 0;
        int score = 0;
        int reward = 0;
        int totalScore = 0;

        if (_globalGradient[round].length == 0) {

            checkGrad(matrix);

            _globalGradient[round] = matrix;

            uint key;
            key = getKey(matrix);
            _globalGradientList[round][key] = matrix;

        } else {
            checkGrad(matrix);


            val = cosineSim(matrix, round);
            val = SafeMath.abs(val);

            if (val == 0) {
                _subCred(1);
            }
            else if (val > 90 && val <= 100) {

                _globalGradient[round] = matrix;

                uint key;
                key = getKey(matrix);
                _globalGradientList[round][key] = matrix;

                _addCred(1); 

                reward = _credReward();
                score = 3;
                totalScore = reward + score;
                _addBalance(uint(totalScore));
            } 
            else if(val > 80 && val <= 90) {

                _globalGradient[round] = matrix;

                uint key;
                key = getKey(matrix);
                _globalGradientList[round][key] = matrix;

                reward = _credReward();
                score = 1;
                totalScore = reward + score;
                _addBalance(uint(totalScore));
            } else {
                _subCred(1);
                score = 0;
            }

        }
 

        return (_cred[msg.sender], totalScore, val);
    }    

    function cosineSim(int[128][] memory newMatrix, uint round) internal view returns (int) {
        // storage to memory
        int256[] memory A_flat = new int256[](_globalGradient[round].length * 128);
        for (uint i = 0; i < _globalGradient[round].length; i++) {
            for (uint j = 0; j < 128; j++) {
                A_flat[i * 128 + j] = _globalGradient[round][i][j];
            }
        }
        int[] memory B_flat = flatten(newMatrix);

        int magA = vectorMag(A_flat);   // 91
        int magB = vectorMag(B_flat);   // 23
        if (magA==0 || magB==0) {
            return 0;
        }
        uint key = uint(magB);
        require(_globalGradientList[round][key].length == 0, "matrix is exist");
        
        int256 dot = dotProduct(A_flat, B_flat);   // 217

        int difVal;
        int difNum;
        (difVal, difNum) = diffVal(A_flat, B_flat);

        if (difNum < 10) {
            return 0;
        }
        if (difVal < 1000000 && difNum < 15) {
            return 0;
        }
        
        return int(dot * 100 / (magA * magB));   // 1.04
    }

    function checkGrad(int[128][] memory grad) internal pure returns (bool) {
        
        int[] memory flat = flatten(grad);  // [1, 2, 3, 4, 5, 6]

        int mag = vectorMag(flat);   // 9
        require(mag!=0, "invalid matrix");
        
        return true;
    }

    function flatten(int[128][] memory matrix) pure internal returns(int[] memory) {
        int[] memory flat = new int[](matrix.length * 128);

        uint index = 0;
        for (uint i = 0; i < matrix.length; i++) {
            for (uint j = 0; j < matrix[i].length; j++) {
                flat[index] = matrix[i][j];
                index++;
            }
        }

        return flat;
    }

    function dotProduct(int[] memory vec1, int[] memory vec2) pure internal returns(int) {
        require(vec1.length == vec2.length, "Arrays must have the same length");
        int dot;
        for(uint i=0; i<vec1.length; i++) {
            dot += vec1[i] * vec2[i];
        }
        return dot;
    }

    function diffVal(int[] memory a, int[] memory b) pure internal returns(int, int) {
        require(a.length == b.length, "Arrays must have the same length");
        int val = 0;
        int dif_num = 0;
        for(uint i=0; i<a.length; i++) {
            if (a[i] != b[i]) {
                dif_num++;
            }
            val += SafeMath.abs(a[i] - b[i]);
        }
        return (val, dif_num);
    }

    function vectorMag(int[] memory vec) pure internal returns(int) {
        int sum = 0;
        for(uint i=0; i<vec.length; i++) {
            sum += vec[i]**2;
        }

        return int(SafeMath.sqrt(uint(sum)));
    }

    function getKey(int[128][] memory matrix) pure internal returns (uint) {
        int[] memory vector = flatten(matrix);
        int mag = vectorMag(vector);
        return uint(mag);
    }

    function getGlobalGradient(uint round) public view returns(int[128][] memory, uint256){
        return (_globalGradient[round], _globalGradient[round].length);
    }

    function getGlobalGradientList(uint round, int[128][] memory matrix) public view returns(int[128][] memory, uint256, uint){
        uint key = getKey(matrix);
        return (_globalGradientList[round][key], _globalGradientList[round][key].length, key);
    }

    function balance(address owner) public view returns (uint256) {
        return _balances[owner];
    }

    function _addBalance(uint256 value) internal {
        _balances[msg.sender] = _balances[msg.sender].add(value);
    }

    function _subBalance(uint256 value) internal {
        _balances[msg.sender] = _balances[msg.sender].sub(value);
    }
    
    function cred(address owner) public view returns (uint256) {
        return _cred[owner];
    }

    function _addCred(uint256 value) internal {
        if (_cred[msg.sender] < 110) {
            _cred[msg.sender] = _cred[msg.sender].add(value);
        }
        
    }

    function _subCred(uint256 value) internal {
        if (_cred[msg.sender] > 0) {
            _cred[msg.sender] = _cred[msg.sender].sub(value);
        }
    }

    function _credReward() internal returns (int){
        if (_cred[msg.sender] >= 110) {
            return 1;
        }
        else if(_cred[msg.sender] < 100) {
            _addCred(1);
            return 0;
        } else {
            return 0;
        }
    }

    function getCred() public view returns (uint256) {
        return _cred[msg.sender];
    }

    function getBalance() public view returns (uint256) {
        return _balances[msg.sender];
    }

}
