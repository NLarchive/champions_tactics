import { Contract, ethers } from "ethers";

async function getABI(contractAddress: string): Promise<any> {
  const apiUrl = "https://explorer.oasys.homeverse.games/api";

  const url = `${apiUrl}?module=contract&action=getabi&address=${contractAddress}`;

  try {
    const response = await fetch(url);
    const data = await response.json();

    if (
      data.status === "1" &&
      data.result !== "Contract source code not verified"
    ) {
      return JSON.parse(data.result);
    } else {
      throw new Error(`Failed to fetch ABI for contract: ${contractAddress}`);
    }
  } catch (error) {
    console.error("Error fetching ABI:", error);
    throw new Error("Failed to fetch ABI");
  }
}

export async function getContract(
  contractAddress: string,
  provider: ethers.JsonRpcProvider,
): Promise<Contract> {
  const abi = await getABI(contractAddress);
  return new ethers.Contract(contractAddress, abi, provider);
}

export async function getNFTs(
  walletAddress: string,
  contractAddress: string,
): Promise<any[]> {
  const provider = new ethers.JsonRpcProvider(
    "https://rpc.mainnet.oasys.homeverse.games/",
  );

  try {
    const contract = await getContract(contractAddress, provider);
    const tokens = await contract.tokensOfOwner(walletAddress);
    if (!tokens.length) {
      return [];
    }
    const NFTs: any[] = [];

    for (const tokenId of tokens) {
      const traitJsonUrl: string = await contract.tokenURI(tokenId.toString());
      try {
        const response = await fetch(traitJsonUrl);
        const data = await response.json();
        NFTs.push(data);
      } catch (e) {
        console.error("Error fetching trait data:", e);
      }
    }

    console.log("NFTs:", NFTs);

    return NFTs;
  } catch (error) {
    console.error("Error fetching NFTs:", error);
    return [];
  }
}
